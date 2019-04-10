#!/bin/bash

SCRIPT=/home/hpc/capm/sn0515/bbDiscriminator/run_cnn.py
SCRIPT_SA=/home/hpc/capm/sn0515/bbDiscriminator/plot_scripts/plot_shape_agreement.py

echo '========================================'
echo 'run Th228 @S5 Check'
echo '========================================'
(python ${SCRIPT} -s Th228 -p S5 -v mc ${@:1})
(python ${SCRIPT} -s Th228 -p S5 -v data ${@:1})
(python ${SCRIPT_SA} -s Th228 -p S5 ${@:1})

echo '========================================'
echo 'run Th228 @S2 Check'
echo '========================================'
(python ${SCRIPT} -s Th228 -p S2 -v mc ${@:1})
(python ${SCRIPT} -s Th228 -p S2 -v data ${@:1})
(python ${SCRIPT_SA} -s Th228 -p S2 ${@:1})

echo '========================================'
echo 'run Th228 @S8 Check'
echo '========================================'
(python ${SCRIPT} -s Th228 -p S8 -v mc ${@:1})
(python ${SCRIPT} -s Th228 -p S8 -v data ${@:1})
(python ${SCRIPT_SA} -s Th228 -p S8 ${@:1})

echo '========================================'
echo 'run Th228 @S11 Check'
echo '========================================'
(python ${SCRIPT} -s Th228 -p S11 -v mc ${@:1})
(python ${SCRIPT} -s Th228 -p S11 -v data ${@:1})
(python ${SCRIPT_SA} -s Th228 -p S11 ${@:1})

echo '========================================'
echo 'run Ra226 @S5 Check'
echo '========================================'
(python ${SCRIPT} -s Ra226 -p S5 -v mc ${@:1})
(python ${SCRIPT} -s Ra226 -p S5 -v data ${@:1})
(python ${SCRIPT_SA} -s Ra226 -p S5 ${@:1})

echo '========================================'
echo 'run Co60 @S5 Check'
echo '========================================'
(python ${SCRIPT} -s Co60 -p S5 -v mc ${@:1})
(python ${SCRIPT} -s Co60 -p S5 -v data ${@:1})
(python ${SCRIPT_SA} -s Co60 -p S5 ${@:1})


echo '==================== Checks finished ===================='
