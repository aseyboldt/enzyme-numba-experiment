all :
	clang test.c -S -emit-llvm -o call_grad.ll -O1 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops
	llvm-link logp_func.ll call_grad.ll -o merged.ll -S
	opt merged.ll -load=../build/Enzyme/LLVMEnzyme-11.so -enzyme -o output.ll -S
	#clang output.ll -O2 -o ./run_O2
	#clang output.ll -O0 -o ./run_O0
	clang -shared -march=native output.ll -ffast-math -O2 -o libgrad.so -fPIC
