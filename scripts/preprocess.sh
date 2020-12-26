set -ex
# command
python3 ../src/pre_post.py \
	--input /home/cz/Downloads/SMPL-AMC-Imitator-master/pose \
	--output /home/cz/cs/PG19/dataset/cmu_poses/poses \
	--mode pre \
	--step 200 \
	--no-norm