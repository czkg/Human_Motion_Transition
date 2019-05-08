import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from itertools import islice


# options
opt = TestOptions().parse()
opt.num_threads = 1    # test code only support num_threads = 1
opt.batch_size = 1     # test code only support batch_size = 1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# test data
for i, data in enumerate(islice(dataset, opt.num_test)):
	model.set_input(data)
	print('process inputs %3.3d/%3.3d' % (i, opt.num_test))