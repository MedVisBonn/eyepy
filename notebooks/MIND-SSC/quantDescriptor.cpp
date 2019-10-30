#include "mex.h"
#include <math.h>
#include <iostream>
#include <inttypes.h>

using namespace std;
#define printf mexPrintf


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	timeval time1,time2;

    float* left=(float*)mxGetData(prhs[0]);
	
	int val=(int)mxGetScalar(prhs[1]);

    const mwSize* dims1=mxGetDimensions(prhs[0]);
	int m=dims1[0]; int n=dims1[1]; int o=dims1[2];
	int d=dims1[3];


	int sz=m*n*o;
	
	int* lefti=new int[sz*d];
	
	for(int i=0;i<sz*d;i++){
		lefti[i]=min(max((int)(left[i]*val-0.5),0),val-1);
	}
	
	uint64_t* tablei=new uint64_t[val]; //intensity values
	for(int i=0;i<val;i++){
		tablei[i]=0ULL;
	}
	uint64_t* tabled=new uint64_t[d]; //descriptor entries
	for(int i=0;i<d;i++){
		tabled[i]=0ULL;
	}
	uint64_t power=1ULL;
	tablei[0]=0;
	for(int i=1;i<val;i++){
		power+=power;
		tablei[i]=power-1ULL;
	}

	tabled[0]=1;
	for(int i=1;i<d;i++){
		tabled[i]=tabled[i-1]*power;

	}
	
	int dimsout[3]={m,n,o};
	
	
    plhs[0]=mxCreateNumericArray(3,dimsout,mxUINT64_CLASS,mxREAL);
	
	uint64_t* left_64=(uint64_t*)mxGetData(plhs[0]);


	for(int i=0;i<sz;i++){
		left_64[i]=0ULL;
		for(int q=0;q<d;q++){
			left_64[i]+=tablei[lefti[i+q*sz]]*tabled[q];
		}
	}

	delete tabled;
	delete tablei;
	

	delete lefti;
	
    return;
}
