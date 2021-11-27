CURDIR=$(readlink -e $(dirname $0))
echo $CURDIR
docker run -v $CURDIR/data:/app/data -v $CURDIR/model:/app/model  nnn/megafon_project  luigi --module src.luigi_pipeline LTaskPredict --local-scheduler $@


