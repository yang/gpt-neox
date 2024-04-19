if [[ $debug ]]
then cmd=debugpy-run
else cmd=python
fi
$cmd deepy.py train.py configs/tiny.yml configs/local_setup.yml
