#include "mex.h"
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <vector>


using namespace std;
#define printf mexPrintf

//usage: dist=hammingMex(ssc_q1,ssc_q2);

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
	timeval time1,time2;
    

    uint64_t* binary1=(uint64_t*)mxGetData(prhs[0]);
    uint64_t* binary2=(uint64_t*)mxGetData(prhs[1]);
    
    const mwSize* dims1=mxGetDimensions(prhs[0]);
    int m=dims1[0]; int n=dims1[1]; int o=dims1[2];
    if(mxGetNumberOfDimensions(prhs[0])==2)
       o=1;
    
    int dims2[]={m,n,o};
    plhs[0]=mxCreateNumericArray(3,dims2,mxSINGLE_CLASS,mxREAL);

	float* result=(float*)mxGetData(plhs[0]);
    
    for(int i=0;i<m*n*o;i++){
        result[i]=0.0f;
    }
    float factor=1.0f/((float)o*60.0f);
    for(int i=0;i<m*n*o;i++){
        result[i]+=__builtin_popcountll(binary1[i]^binary2[i])*factor;
    }
    
    
    
}
