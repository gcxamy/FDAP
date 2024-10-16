Firstly you can create an environment follow this link:
Then to attack the images, you can :
python our_attacks.py --attack attack_name --gpu 0 --batch_size 1 --model_name deit_base_distilled_patch16_224 --attention_layer_start 5 --attention_layer_end 9
To evaluate the attack, refer to evaluate.sh
