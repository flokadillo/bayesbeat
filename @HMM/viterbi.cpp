// MEX function to compute viterbi loop over states
// [map_state_sequence] = viterbi(state_ids_i, state_ids_j, trans_prob_ij, ...
//              initial_prob, obs_lik, state_2_r_pos)
//
// WARNING: assumes that state_ids_j of the transition model are sorted!
//  TODO: Check if that's true
//
// INPUT:
//      state_ids_i=[1:4]';
//      state_ids_j=[2;1;4;3];
//      trans_prob_ij=[1;1;1;1];
//      initial_prob=[0.5;0;0.5;0];
//      obs_lik=zeros(2, 2, 4);
//      obs_lik(1, :, :)=[0.7, 0.2, 0.7, 0.9; 0.3, 0.8, 0.3, 0.1];
//      obs_lik(2, :, :)=[0.5, 0.4, 0.7, 0.5; 0.5, 0.6, 0.3, 0.5];
//      obs_lik = [0.7, 0.1, 0.2, 0.1; 0.3, 0.9, 0.8, 0.9];
//      gmm_from_state = [1; 2; 2; 1]
//      state_2_r_pos=[1, 1; 1, 2; 2, 1; 2, 2];
//
// 07.05.2015 by Florian Krebs
// 17.06.2015 integrated uint16/uint32 switch by Harald Frostel
// ---------------------------------------------------------------------

#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <algorithm>
#include <vector>
#include <stdint.h>
/* Definitions to keep compatibility with earlier versions of ML */
#ifndef MWSIZE_MAX
typedef int mwSize;
typedef int mwIndex;
typedef int mwSignedIndex;

#if (defined(_LP64) || defined(_WIN64)) && !defined(MX_COMPAT_32)
/* Currently 2^48 based on hardware limitations */
# define MWSIZE_MAX    281474976710655UL
# define MWINDEX_MAX   281474976710655UL
# define MWSINDEX_MAX  281474976710655L
# define MWSINDEX_MIN -281474976710655L
#else
# define MWSIZE_MAX    2147483647UL
# define MWINDEX_MAX   2147483647UL
# define MWSINDEX_MAX  2147483647L
# define MWSINDEX_MIN -2147483647L
#endif
#define MWSIZE_MIN    0UL
#define MWINDEX_MIN   0UL
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 6) {
        mexErrMsgTxt("To few arguments.");
    }
    if (nrhs > 6) {
        mexErrMsgTxt("To many arguments.");
    }
    
//declare variables
    mxArray *map_state_sequence, *psi;
    mxArray *debug_data;
    const mwSize *dims;
    double R, sum_k, temp, debug_temp, check;
    int num_frames, num_trans;
    int p, r, i, j, i_trans, prev_end_state, current_start_state, current_end_state;
    int i_state;
    
//associate inputs and associate pointers
    double *state_ids_i_ptr = mxGetPr(prhs[0]);
    double *state_ids_j_ptr = mxGetPr(prhs[1]);
    double *trans_prob_ij_ptr = mxGetPr(prhs[2]);
    double *initial_prob_ptr = mxGetPr(prhs[3]);
    double *obs_lik_ptr = mxGetPr(prhs[4]);
    double *state_2_r_pos_ptr = mxGetPr(prhs[5]);
    double *gmm_from_state_ptr = mxGetPr(prhs[5]);
    
//figure out dimensions
    // num_frames
    dims = mxGetDimensions(prhs[4]);
    num_frames = (int)dims[1];
    // num_states
    dims = mxGetDimensions(prhs[3]);
    const mwSize num_states = dims[0];
    // number of possible transitions
    dims = mxGetDimensions(prhs[1]);
    num_trans = (int)dims[0];
    
//associate outputs
    map_state_sequence = plhs[0] = mxCreateDoubleMatrix(num_frames,1,mxREAL);
//   debug_data = plhs[1] = mxCreateDoubleMatrix(num_states, 1, mxREAL);
    
//internal variables
    // if less than 65535 states are used we can store the state ids as uint16
    const bool elems32 = (num_states) > 65535;
    uint16_t *psi_ptr_16 = NULL;
    uint32_t *psi_ptr_32 = NULL;
    if (elems32) {
        psi = mxCreateNumericMatrix(num_states, num_frames, mxUINT32_CLASS, mxREAL);
        psi_ptr_32 = static_cast<uint32_t *>(mxGetData(psi));
    } else {
        psi = mxCreateNumericMatrix(num_states, num_frames, mxUINT16_CLASS, mxREAL);
        psi_ptr_16 = static_cast<uint16_t *>(mxGetData(psi));
    }
    std::vector<double> delta(initial_prob_ptr, initial_prob_ptr+num_states);
    std::vector<double> prediction(initial_prob_ptr, initial_prob_ptr+num_states);
    
//associate pointers (get pointer to the first element of the real data)
    double *map_state_sequence_ptr = mxGetPr(map_state_sequence);
    
//start computing
    debug_temp = 0;
    for(i=0;i<num_frames;i++)
    {
        prev_end_state = -7;
        // loop over possible transitions
        for(i_trans=0;i_trans<num_trans;i_trans++)
        {
            // get start and end state of transition i_trans
            current_start_state = (int)state_ids_i_ptr[i_trans]-1;
            current_end_state = (int)state_ids_j_ptr[i_trans]-1;
            if (current_end_state == prev_end_state) {
                // the transition i_trans-1 has the same end_state as transition
                // i_trans: Find the best start_state among these
                temp = delta[current_start_state] * trans_prob_ij_ptr[i_trans];
                if ( temp > prediction[current_end_state]) {
                    // found more probable precursor state -> save it
                    prediction[current_end_state] = temp;
                    const size_t idx = current_end_state * num_frames + i;
                    if (elems32) {
                        psi_ptr_32[idx] = current_start_state;
                    }
                    else {
                        psi_ptr_16[idx] = current_start_state;
                    }
                }
            } else {
                // the transition i_trans has a different end-state from transition
                // i_trans-1
                prev_end_state = current_end_state;
                prediction[current_end_state] = delta[current_start_state] * trans_prob_ij_ptr[i_trans];
                const size_t idx = current_end_state * num_frames + i;
                if (elems32) {
                    psi_ptr_32[idx] = current_start_state;
                }
                else {
                    psi_ptr_16[idx] = current_start_state;
                }
            }
        }
        sum_k = 0;
        for (i_state=0; i_state<num_states; i_state++) {
            // multiply with observation likelihood and sum up
            delta[i_state] = prediction[i_state] * obs_lik_ptr[(int)gmm_from_state_ptr[i_state]];
            sum_k += delta[i_state];
        }
        // normalise
        for (i_state=0; i_state<num_states; i_state++) {
            delta[i_state] /= sum_k;
        }
    }
    // Back tracing
    // Find best_end_state
    int best_end_state = 0;
    double best_delta = -7;
    for (i_state=0; i_state<num_states; i_state++) {
        if (delta[i_state] > best_delta) {
            best_delta = delta[i_state];
            best_end_state = i_state;
        }
    }
    // store and convert to MATLAB index
    map_state_sequence_ptr[num_frames-1] = (double)best_end_state + 1;
    for (i=num_frames-1; i>0; i--) {
        const size_t idx = (best_end_state * num_frames + i);  // idx for psi
        const uint32_t value = elems32?psi_ptr_32[idx]:uint32_t(psi_ptr_16[idx]);
        map_state_sequence_ptr[i-1] = value + 1;
        best_end_state = value;
    }
    mxDestroyArray(psi);
}