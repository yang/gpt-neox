if [[ ! $par ]]
then par=1
fi

if [[ $par == 1 ]]
then export CUDA_VISIBLE_DEVICES=0
fi

case $size in
s ) config=125M ;;
m ) config=6-7B ;;
l ) config=20B ;;
esac

if [[ ! $ds ]]
then ds=0
fi

if [[ $ds == 1 ]]
then dsconfig=configs/dsinf.yml
fi

python deepy.py generate.py configs/$config-gen.yml $dsconfig configs/local_setup.yml configs/text_generation.yml
mv sample_output.txt sample_output_config-$config-ds-$ds-par-$par.txt
