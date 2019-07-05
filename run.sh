#xelab work.test_dsam_encoder -prj work.prj -debug all
xelab work.dsam_encoder_tb_1 -prj work.prj -debug all
#xsim work.test_dsam_encoder -R -wdb wave.wdb
xsim work.dsam_encoder_tb_1 -R -wdb wave.wdb
#xsim work.test_dsam_encoder -g -view wave.wdb 
