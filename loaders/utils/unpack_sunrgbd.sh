# == Move this to the data folder which contains your zipped SUN_RGB data ==
# Create the directories.
mkdir -p rgb/train rgb/test depth/train depth/test semantic37/train semantic37/test semantic13/train semantic13/test
# Unpack the RGB images.
tar -xzf SUNRGBD-test_images.tgz -C rgb/test
tar -xzf SUNRGBD-train_images.tgz -C rgb/train
# Unpack the semantic 37 class labels and move them to the correct directory.
mkdir -p semantic37/train semantic37/test
tar -xzf sunrgbd_train_test_labels.tar.gz
mv img-00[0-4]*.png semantic37/test && mv img-0050[0-4]*.png semantic37/test && mv img-005050.png semantic37/test
mv img-*.png semantic37/train
# Unpack the depth images.
tar -xzf sunrgb_train_depth.tgz -C depth/train &
tar -xzf sunrgb_test_depth.tgz -C depth/test
# Unpack the semantic 13 class labels.
tar -xzf test13labels.tgz -C semantic13/test &
tar -xzf train13labels.tgz -C semantic13/train
echo "Done!"
