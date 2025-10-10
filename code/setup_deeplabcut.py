import os

# !!!
# UPDATE THIS WITH OUTPUT DIRECTORY
# !!!
outdir = "/mnt/home/wslatton/ceph/reprod_population_geom_opt_coding/results/"

current_file_path = os.path.abspath(__file__)
project_root_path = os.path.dirname(os.path.dirname(current_file_path))

import subprocess as sp

if not os.path.exists(outdir):
    os.makedirs(outdir)

# STEP 1: download benchmark dataset if not already present
if not os.path.exists(os.path.join(outdir, "dataset")):
    print("Downloading benchmark dataset...")
    sp.run(
        ["wget", "-q", "https://zenodo.org/records/5851109/files/pups-dlc-2021-03-24.zip?download=1"],
        cwd=outdir
    )
    sp.run(
        ["unzip", "pups-dlc-2021-03-24.zip?download=1", "-d", "dataset"],
        cwd=outdir
    )

# STEP 2: set up training split if not already done

import deeplabcut
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import auxiliaryfunctions
import deeplabcut.pose_estimation_tensorflow.core.predict as predict
import pandas as pd

config_path = os.path.join(outdir, "dataset", "pups-dlc-2021-03-24", "config.yaml")

if not os.path.exists(os.path.join(outdir, "dataset", "pups-dlc-2021-03-24", "training-datasets")):
    print("Creating train/test split...")
    deeplabcut.create_training_dataset(
        config_path
    )

# STEP 3: train the network
weights_path = os.path.join(outdir, "dataset", "pups-dlc-2021-03-24", "dlc-models", "iteration-0", "pupsMar24-trainset70shuffle1", "train", "snapshot-30000")

if not os.path.exists(weights_path + ".index"):
    print("Training network...")
    deeplabcut.train_network(
        config_path,
        displayiters=100, saveiteres=1000, maxiters=30000
    )
    print("Training done.")

# STEP 4: load trained network and training data
cfg = auxiliaryfunctions.read_config(
    config_path
)
test_pose_cfg = load_config(
    os.path.join(outdir, "dataset", "pups-dlc-2021-03-24", "dlc-models", "iteration-0", "pupsMar24-trainset70shuffle1", "test", "pose_cfg.yaml")
)
test_pose_cfg["init_weights"] = weights_path
scale = test_pose_cfg["global_scale"]
sess, inputs, outputs = predict.setup_pose_prediction(
    test_pose_cfg
)
Data = pd.read_hdf(
    os.path.join(
        cfg["project_path"],
        "training-datasets/iteration-0/UnaugmentedDataSet_pupsMar24/"
        "CollectedData_" + cfg["scorer"] + ".h5",
    )
)[cfg["scorer"]]
Data = Data.xs("single", level=0, axis=1).dropna()

# generate sanity check image for pose estimation network
from PIL import Image, ImageDraw
from deeplabcut.utils.auxfun_videos import imread, imresize
import numpy as np

ix = 38 # index of training frame to test
print(ix)
imagename = Data.index[ix]
image = imread(
    os.path.join(cfg["project_path"], *imagename),
    mode="skimage",
)
if scale != 1:
    image = imresize(image, scale)

image = np.array(Image.fromarray(image).resize((563, 384)))[np.newaxis]

# Compute prediction with the CNN
outputs_np = sess.run(
    outputs, feed_dict={inputs: image}
)
scmap, locref = predict.extract_cnn_output(
    outputs_np, test_pose_cfg
)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(
    scmap, locref, test_pose_cfg["stride"]
)

# ground-truth pose
xs_gt = scale * Data.iloc[ix, :].xs("x", level=1).values
ys_gt = scale * Data.iloc[ix, :].xs("y", level=1).values

image = Image.fromarray(image[0])
draw = ImageDraw.Draw(image)

for i in range(len(xs_gt)):
    x = int(xs_gt[i])
    y = int(ys_gt[i])
    draw.ellipse([(x - 5, y - 5), (x + 3, y + 3)], fill="#ff0000")

for i in range(5, 17):
    x = int(pose[i, 0])
    y = int(pose[i, 1])
    draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill="#4dd668")

image.save(os.path.join(outdir, "sanity_check_pose_estimation.png"))

# STEP 5: sample projections of internal network activations (can be quite slow)
from tqdm import tqdm
import numpy.random as npr
import pickle

# list of intermediate tensors to log
target_tensors = {}

for block_ix, block_size in enumerate([3, 4, 6, 3]):
    for unit_ix in range(block_size):
        # residual unit
        label = f"b{block_ix + 1}u{unit_ix + 1}_add"
        full_name = f"resnet_v1_50/block{block_ix + 1}/unit_{unit_ix + 1}/bottleneck_v1/add:0"
        target_tensors[label] = full_name

        # post-activation unit
        label = f"b{block_ix + 1}u{unit_ix + 1}_relu"
        full_name = f"resnet_v1_50/block{block_ix + 1}/unit_{unit_ix + 1}/bottleneck_v1/Relu:0"
        target_tensors[label] = full_name

# trained deconvolutional head for pose estimation
target_tensors["pred"] = "pose/part_pred/block4/BiasAdd:0"

# random projection down to N dimensions for each layer
N = 100

for proj_ix in range(0, 20):
    save_path = os.path.join(outdir, f"xs_all_{proj_ix}.pkl")

    if os.path.exists(save_path):
        print(f"Projection {proj_ix + 1} already exists, skipping...")
        continue

    print(f"Starting projection {proj_ix + 1}...")
    random_projections = None

    # saved intermediate representations
    xs_all = {
        label: np.zeros((len(Data), N))
        for label in target_tensors.keys()
    }

    for ix in tqdm(range(len(Data))):
        imagename = Data.index[ix]
        image = imread(
            os.path.join(cfg["project_path"], *imagename),
            mode="skimage",
        )
        if scale != 1:
            image = imresize(image, scale)

        image = np.array(Image.fromarray(image).resize((563, 384)))[np.newaxis]
        outputs_np = sess.run(
            list(target_tensors.values()), feed_dict={inputs: image}
        )

        if random_projections is None:
            random_projections = []
            print("> Initializing random projections...")
            # first iteration, initialize random projections
            for i, tensor in enumerate(outputs_np):
                random_projections.append(
                    npr.normal(size=(N, tensor.size)) / np.sqrt(tensor.size)
                )
            print("> Done.")

        for label, tensor, proj in zip(target_tensors.keys(), outputs_np, random_projections):
            xs_all[label][ix] = proj @ tensor.ravel()
        
    pickle.dump(xs_all, open(save_path, "wb"))

# STEP 6: assemble all random projections and latents into a pickle file
import glob

# latents (ground-truth pose coordinates)
zs = Data.values
zs -= np.mean(zs, axis=0)

xs_alls = {
    label: []
    for label in target_tensors.keys()
}

proj_ix = 0
rep_files = glob.glob(os.path.join(outdir, "xs_all_*.pkl"))

for file in rep_files:
    xs_all = pickle.load(open(file, "rb"))

    # center representations
    xs_all = {
        label: xs - np.mean(xs, axis=0)
        for label, xs in xs_all.items()
    }

    for label in xs_all.keys():
        xs_alls[label].append(xs_all[label])

pickle.dump((zs, xs_alls), open(os.path.join(outdir, "dlc_reps.pkl"), "wb"))
print("Done!")