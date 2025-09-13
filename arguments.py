import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process Reward Optimization.")
    
    # paths
    parser.add_argument("--setting", type=str, help="Setting being generated", default="test")
    parser.add_argument("--cache_dir",type=str,help="HF cache directory",default="./cache")
    parser.add_argument("--save_dir",type=str,help="Directory to save images",default="./outputs")
    
    # model and optim
    parser.add_argument("--model", type=str, help="Model to use", default="sd-turbo")
    parser.add_argument("--lr", type=float, help="Learning rate", default=5.0)
    parser.add_argument("--n_iters", type=int, help="Number of iterations", default=50)
    parser.add_argument("--n_inference_steps", type=int, help="Number of iterations", default=1)
    parser.add_argument("--optim",choices=["sgd", "adam", "lbfgs"],default="sgd",help="Optimizer to be used")
    parser.add_argument("--nesterov", default=True, action="store_false")
    parser.add_argument("--grad_clip", type=float, help="Gradient clipping", default=0.1)
    parser.add_argument("--seed", type=int, help="Seed to use", default=0)

    # Rewards and Weighting
    parser.add_argument("--disable_hps", default=True, action="store_false", dest="enable_hps")
    parser.add_argument("--disable_imagereward",default=True,action="store_false",dest="enable_imagereward")
    parser.add_argument("--disable_vqa_score",default=True,action="store_false",dest="enable_vqa")
    parser.add_argument("--disable_da_score",default=True,action="store_false",dest="enable_da")
    parser.add_argument("--disable_reg", default=True, action="store_false", dest="enable_reg")

    parser.add_argument("--hps_weight", type=float, help="Weighting for HPS", default= 5.0)
    parser.add_argument("--imagereward_weight",type=float,help="Weighting for ImageReward",default=5.0)
    parser.add_argument("--vqa_weight",type=float,help="Weighting for VQA score",default=1.0)
    parser.add_argument("--da_weight",type=float,help="Weighting for DA score",default=5.0)
    parser.add_argument("--reg_weight", type=float, help="Regularization weight", default=0.01)

    # asset paths 
    parser.add_argument("--prompt_file", type=str, help="Name of the prompt set in assets", default="example_prompts")
    parser.add_argument("--category_file",  type=str, help="Name of the category guess set in assets", default="category_guess")
    parser.add_argument("--adaptive_weights_file",  type=str, help="Name of the category guess set in assets", default="weights")

    # general
    parser.add_argument("--imageselect", default=False, action="store_true")
    parser.add_argument("--save_all_images", default=False, action="store_true")
    parser.add_argument("--save_every_10_image", default=False, action="store_true")
    parser.add_argument("--save_every_5_image", default=False, action="store_true")
    parser.add_argument("--no_optim", default=False, action="store_true")
    parser.add_argument("--dtype", type=str, help="Data type to use", default="float16")
    parser.add_argument("--device_id", type=str, help="Device ID to use", default=None)
    parser.add_argument("--k", default=1, help="primary seed")
    # for example, while running seed = 5, save the result for seed = 3
    parser.add_argument("--x", default=1, help="secondary seed")

    # optional multi-step model
    parser.add_argument("--enable_multi_apply", default=False, action="store_true")
    parser.add_argument("--multi_step_model", type=str, help="Model to use", default="flux")
    
    args = parser.parse_args()
    return args
