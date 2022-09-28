export PYTHONPATH=/home/guohao/.local/lib
for adaption_type in "finetune"
do 
    for domain in  "poetry" "international" "sports" "story" 
    # for domain in "gongwen"
    #poetry data too short, need to fix
    do
        for shotnum in 0
        do
        python3.8 test.py --shotnum $shotnum --domain $domain --adaption_type $adaption_type
        done
    done
done
