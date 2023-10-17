

def create_opt():
	import AugmentStrat.CATGAN_strat.config as opt
	return opt
# opt = parser.parse_args()
#
# 	if opt.if_real_data:
# 		opt.max_seq_len, opt.vocab_size = text_process(opt.path+'/CatGAN/dataset/' + opt.dataset + '.txt')
# 		cfg.extend_vocab_size = len(load_test_dict(opt.dataset, opt.path)[0])  # init classifier vocab_size
# 	cfg.init_param(opt, opt.path)
# 	opt.save_root = cfg.save_root
# 	opt.train_data = cfg.train_data
# 	opt.test_data = cfg.test_data