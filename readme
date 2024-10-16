

# Environment
Recover the environment by
```
conda env create -f environment_transformer.yml
```



# Models
ViTs models from [timm]: 
* deit_base_patch16_224
* deit_base_distilled_patch16_224
* cait_s24_224
* convit_base  


# Implementation
Change **ROOT_PATH** of utils.py.
### attack
```
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name deit_base_patch16_224 --attention_layer_start 5 --attention_layer_end 9
```
* attack: the attack method, OurAlgorithm, OurAlgorithm_MI or OurAlgorithm_SGM
* model_name: white-box model name, deit_distilled_base_patch16_224,deit_distilled_base_patch16_224, cait_s24_224, convit_base
* attention_layer_start: start layer of attention
* attention_layer_end: end layer of attention
### evaluate
```
refer to run_evaluate.sh to evaluate the attack
```

```
