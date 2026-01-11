import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from huggingface_hub import login
import os

# ==========================================
# 0. Auth
# ==========================================
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except:
    HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 直接入力用

if HF_TOKEN and not HF_TOKEN.startswith("hf_xxxx"):
    login(token=HF_TOKEN, add_to_git_credential=False)

# ==========================================
# 1. Config & Logic (With Calibration)
# ==========================================
@dataclass
class SIAConfig:
    tau: float = 0.45
    # Omegaは自動決定するため初期値はダミー
    omega: float = 0.0      
    k_future: float = 5.0
    decay: float = 0.99
    enable_intervention: bool = True
    max_reject_attempts: int = 5
    
    # Calibration設定
    calibration_steps: int = 15  # 最初の何ステップで基準を決めるか
    sigma_threshold: float = 3.0 # 平均 + 3σ を閾値にする

class SIAMonitor:
    def __init__(self, config: SIAConfig, hidden_dim: int):
        self.cfg = config
        self.hidden_dim = hidden_dim
        self.exp_nll = 0.0
        self.scar_vec = None
        self.scar_norm = 0.0
        
        # Calibration用バッファ
        self.calibration_shocks = []
        self.is_calibrated = False
        
        self.logs = {
            "tokens": [], "nll": [], "event": [], "iso": [], 
            "shock": [], "rejected": []
        }

    def _normalize(self, vec):
        return F.normalize(vec, p=2, dim=-1)

    def compute_shock(self, nll_scalar: float, current_hidden: torch.Tensor):
        h_norm = self._normalize(current_hidden)
        
        # Event
        diff = abs(nll_scalar - self.exp_nll)
        event = max(0.0, diff)

        # Isomorphism
        I_t = 0.0
        if self.scar_vec is not None and self.scar_norm > 1e-6:
            if self.scar_vec.device != h_norm.device:
                self.scar_vec = self.scar_vec.to(h_norm.device)
            sim = torch.dot(h_norm, self.scar_vec).item()
            I_t = max(0.0, sim)

        shock = event * (1.0 + self.cfg.k_future * I_t * self.scar_norm)
        return shock, event, I_t

    def update(self, nll_scalar: float, current_hidden: torch.Tensor, event: float):
        h_norm = self._normalize(current_hidden).detach()
        self.exp_nll = (1 - self.cfg.tau) * self.exp_nll + self.cfg.tau * nll_scalar
        
        if self.scar_vec is None:
            self.scar_vec = torch.zeros_like(h_norm)
        if self.scar_vec.device != h_norm.device:
            self.scar_vec = self.scar_vec.to(h_norm.device)

        self.scar_vec = self.cfg.decay * self.scar_vec + event * h_norm
        self.scar_norm = self.scar_vec.norm().item()

    def calibrate(self):
        """蓄積したShockデータからOmegaを決定する"""
        if not self.calibration_shocks:
            return
        
        shocks = np.array(self.calibration_shocks)
        mu = np.mean(shocks)
        sigma = np.std(shocks)
        
        # Omega = Mean + 3 * Sigma (統計的異常値)
        # ただし最低値(floor)を設けて、静かすぎる時の過敏反応を防ぐ
        calculated_omega = mu + self.cfg.sigma_threshold * sigma
        self.cfg.omega = max(calculated_omega, 5.0) 
        
        self.is_calibrated = True
        print(f"\n[Calibration Done] Shock Mean={mu:.2f}, Std={sigma:.2f} -> New Omega={self.cfg.omega:.2f}")

    def log_step(self, token_str, nll, event, iso, shock, is_rejected=0):
        self.logs["tokens"].append(token_str)
        self.logs["nll"].append(nll)
        self.logs["event"].append(event)
        self.logs["iso"].append(iso)
        self.logs["shock"].append(shock)
        self.logs["rejected"].append(is_rejected)

# ==========================================
# 2. Loader
# ==========================================
def load_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

# ==========================================
# 3. Generation Loop (Auto-Calibration)
# ==========================================
def generate_calibrated(model, tokenizer, prompt, sia_config, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    curr_ids = inputs.input_ids
    past_key_values = None
    hidden_dim = model.config.hidden_size
    sia = SIAMonitor(sia_config, hidden_dim)
    
    print(f"Prompt: {prompt}")
    print("Generating (Calibrating first...): ", end="")
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=curr_ids, 
                past_key_values=past_key_values, 
                output_hidden_states=True,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]
            last_hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
            probs_all = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs_all, descending=True, dim=-1)
            
            # --- Phase判定 ---
            is_calibrating = (step < sia_config.calibration_steps)
            
            if step == sia_config.calibration_steps:
                sia.calibrate() # ここでOmegaが確定する

            # --- Candidate Selection ---
            chosen_token_id = None
            chosen_data = {}
            reject_count = 0
            
            # 候補探索
            for i in range(min(sia_config.max_reject_attempts + 1, sorted_indices.size(1))):
                candidate_id = sorted_indices[0, i]
                candidate_prob = sorted_probs[0, i].item()
                nll = -np.log(candidate_prob + 1e-9)
                shock, event, iso = sia.compute_shock(nll, last_hidden)
                
                # 介入判定: 
                # 1. キャリブレーション完了後であること
                # 2. 介入有効であること
                # 3. Shock > Omega であること
                if (not is_calibrating) and sia_config.enable_intervention and (shock > sia.cfg.omega):
                    reject_count += 1
                    continue
                else:
                    # 採用
                    chosen_token_id = candidate_id
                    chosen_data = {"nll": nll, "shock": shock, "event": event, "iso": iso}
                    break
            
            # Fallback
            if chosen_token_id is None:
                chosen_token_id = sorted_indices[0, 0]
                prob = sorted_probs[0, 0].item()
                nll = -np.log(prob + 1e-9)
                shock, event, iso = sia.compute_shock(nll, last_hidden)
                chosen_data = {"nll": nll, "shock": shock, "event": event, "iso": iso}

            # Calibration中はShockデータを収集
            if is_calibrating:
                sia.calibration_shocks.append(chosen_data["shock"])

            token_str = tokenizer.decode(chosen_token_id)
            print(token_str, end="", flush=True)
            
            is_rejected_flag = 1.0 if reject_count > 0 else 0.0
            sia.log_step(token_str, chosen_data["nll"], chosen_data["event"], chosen_data["iso"], chosen_data["shock"], is_rejected_flag)
            sia.update(chosen_data["nll"], last_hidden, chosen_data["event"])
            
            curr_ids = chosen_token_id.unsqueeze(0).unsqueeze(0)
            past_key_values = outputs.past_key_values
            
            if chosen_token_id.item() == tokenizer.eos_token_id:
                break
    print("\nDone.")
    return sia

def plot_results(sia, model_name, filename="sia_calibrated_result.png"):
    tokens = sia.logs["tokens"]
    x = range(len(tokens))
    fig, ax = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    # Calibrationラインの表示
    calib_x = sia.cfg.calibration_steps
    
    ax[0].plot(x, sia.logs["nll"], label="NLL", color="blue")
    ax[0].axvline(x=calib_x, color="gray", linestyle=":", label="Calib End")
    ax[0].set_title(f"SIA Auto-Calibrated (Omega={sia.cfg.omega:.1f})")
    ax[0].legend(loc="upper right")
    
    ax[1].plot(x, sia.logs["iso"], label="Isomorphism", color="green")
    ax[1].axvline(x=calib_x, color="gray", linestyle=":")
    
    ax[2].plot(x, sia.logs["shock"], label="Shock", color="red")
    ax[2].axhline(y=sia.cfg.omega, color="black", linestyle="--", label="Auto Omega")
    ax[2].axvline(x=calib_x, color="gray", linestyle=":")
    ax[2].legend(loc="upper right")
    
    rejects = np.array(sia.logs["rejected"])
    ax[3].bar(x, rejects, color="purple", label="Intervention", alpha=0.7)
    ax[3].axvline(x=calib_x, color="gray", linestyle=":")
    
    step = max(1, len(tokens) // 50)
    plt.xticks(x[::step], tokens[::step], rotation=90, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    PROMPT = "Alice is faster than Bob. Bob is faster than Charlie. Charlie is faster than Alice. Therefore, Alice is"
    
    model, tokenizer = load_model(MODEL_NAME)
    
    # Omegaは0.0でスタートし、動的に決定される
    config = SIAConfig(tau=0.45, omega=0.0, k_future=5.0, calibration_steps=15, sigma_threshold=3.0)
    
    sia = generate_calibrated(model, tokenizer, PROMPT, config, max_new_tokens=60)
    plot_results(sia, MODEL_NAME)
