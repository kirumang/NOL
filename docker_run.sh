image_name=nol:latest
#source_dir=$(cd $(dirname $0) && pwd)
data_mount_arg="-v /home/kiru/media/hdd_linux:/root/ext"
xhost +local:root;
docker run --runtime nvidia --net=host -it -e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw $data_mount_arg $image_name
xhost -local:root;
