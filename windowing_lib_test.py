import windowing_library.py

def func(x):
	return x+1

def test_answer():
	x = [1,2,3,4,5,6,7,8,9,10]
	y = [1,2,3,4,5,6,7,8,9,10]
	z = [1,2,3,4,5,6,7,8,9,10]
	nanos = [1,2,3,4,5,6,7,8,9,10]
	sample = [x,y,z,nanos]
	windows= [x,y,z,nanos]
	assert initial_find_lift(sample, 4, 2, 1)