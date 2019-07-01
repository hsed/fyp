# cloud-init

```bash
cat /dev/zero | ssh-keygen -q -N ""
cat ~/.ssh/id_rsa.pub
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
# <after adding key to git>
git clone git@github.com:hsed/fyp.git
cd fyp

apt update && apt install -y libsm6 libxext6 libxrender-dev zip unzip curl wget nano
pip install jupyterlab tensorflow tensorboardx opencv-python h5py seaborn pyyaml==5.1

curl -L https://imperialcollegelondon.box.com/shared/static/LINK_HIDDEN.zip -o datasets/hand_pose_action/data_train_hpe_cache.h5
curl -L https://imperialcollegelondon.box.com/shared/static/LINK_HIDDEN.zip -o datasets/hand_pose_action/data_test_hpe_cache.h5
curl -L https://imperialcollegelondon.box.com/shared/static/LINK_HIDDEN.zip -o datasets/hand_pose_action/hand_pose_ann_v1.zip
curl -L https://imperialcollegelondon.box.com/shared/static/LINK_HIDDEN.zip -o datasets/hand_pose_action/dataset_cache.zip
curl -L https://imperialcollegelondon.box.com/shared/static/LINK_HIDDEN.zip -o datasets/hand_pose_action/hpe_dataset_cache.zip

cd datasets/hand_pose_action
unzip hand_pose_ann_v1.zip
unzip dataset_cache.zip
unzip hpe_dataset_cache.zip
```
