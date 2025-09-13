# **CARINOX**

## **Setup**
To install the necessary dependencies, use the provided `environment.yml` file. Additionally, run the `hps_installer.py` script to acquire any required assets.

### **Installation**
```bash
conda env create -f environment.yml
conda activate carinox
python hps_installer.py
```
Ensure that all required models and dependencies are correctly installed.

## **Usage**
Run `main.py` with various configurations:

```bash
python main.py --model {"sd-turbo" | "sdxl-turbo" | "pixart"} \
               --k <int> \
               [--disable_vqa] [--disable_da] [--disable_hps] [--disable_imagereward] \
               --prompt_file <name_of_prompt_file_in_assets> \
               [--save_all_images] [--no_optim] [--not_adaptive] \
               [--cache_dir <path>] [--save_dir <path>] \
               [--lr <float>] [--n_iters <int>] [--n_inference_steps <int>] \
               [--optim {sgd | adam | lbfgs}] [--grad_clip <float>] \
               [--hps_weight <float>] [--imagereward_weight <float>] \
               [--vqa_weight <float>] [--da_weight <float>] \
               [--reg_weight <float>]
```

### **Example Commands**

#### Basic Image Generation:
```bash
python main.py --model "sd-turbo" --prompt_file "example_prompts" --save_all_images
```

#### Multi-Step Optimization with PixArt Model:
```bash
python main.py --model "pixart" --k 3 --lr 5 --save_all_images
```

#### Disable HPS & VQA, Optimize SDXL-Turbo Model with Custom Weighting:
```bash
python main.py --model "sdxl-turbo" \
               --disable_da --disable_imagereward \
               --hps_weight 3.0 --vqa_weight 0.5 \
```


