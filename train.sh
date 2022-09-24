export PYTHONPATH=/home/guohao/.local/lib
for adaption_type in "finetune"
do 
    for domain in "poetry" "international" "sports" "story"
    #poetry data too short, need to fix
    do
        for shotnum in 4 8 16 32 64 128
        do
        python3.8 train.py --shotnum $shotnum --domain $domain --adaption_type $adaption_type
        done
    done
done
