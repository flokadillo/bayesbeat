// MEX function to compute viterbi loop over states
// [map_state_sequence] = viterbi(state_ids_i, state_ids_j, trans_prob_ij, ...
//              initial_prob, obs_lik, state_2_r_pos, valid_states, validstate_to_state)
//
// WARNING: assumes that state_ids_j of the transition model are sorted! 
//  TODO: Check if that's true
//
// INPUT:
//      state_ids_i=[1:4]'; state_ids_j=[2;1;4;3]; trans_prob_ij=[1;1;1;1]; initial_prob=[0.5;0;0.5;0]; obs_lik=zeros(2, 2, 4); obs_lik(1, :, :)=[0.7, 0.2, 0.7, 0.9; 0.3, 0.8, 0.3, 0.1]; obs_lik(2, :, :)=[0.5, 0.4, 0.7, 0.5; 0.5, 0.6, 0.3, 0.5]; state_2_r_pos=[1, 1; 1, 2; 2, 1; 2, 2]; validstate_to_state=unique(state_ids_j); valid_states=zeros(max(state_ids_j), 1); valid_states(validstate_to_state)=1:length(validstate_to_state);
//
//
//
//
//
//

#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <algorithm>
#include <vector>
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
  if (nrhs < 8) {
        mexErrMsgTxt("To few arguments.");
  }
  if (nrhs > 8) {
        mexErrMsgTxt("To many arguments.");
  } 
  
//declare variables
  mxArray *map_state_sequence, *psi;
  mxArray *debug_data;
  const mwSize *dims;
  double sum_k, temp;
  mwSignedIndex *psi_ptr;
  int num_frames, num_states, obs_idx, R, num_pos, num_trans, num_valid_states;
  int i, j, r, p, i_trans, prev_end_state, current_start_state, current_end_state;
  int idx, i_state, last_state_valid;
  
//associate inputs and associate pointers
  double *state_ids_i_ptr = mxGetPr(prhs[0]);
  double *state_ids_j_ptr = mxGetPr(prhs[1]);
  double *trans_prob_ij_ptr = mxGetPr(prhs[2]);
  double *initial_prob_ptr = mxGetPr(prhs[3]);
  double *obs_lik_ptr = mxGetPr(prhs[4]);
  double *state_2_r_pos_ptr = mxGetPr(prhs[5]);
  double *valid_states_ptr = mxGetPr(prhs[6]);
  double *validstate_to_state_ptr = mxGetPr(prhs[7]);
   
//figure out dimensions
  // num_frames
  dims = mxGetDimensions(prhs[4]);
  num_frames = (int)dims[2];
  dims = mxGetDimensions(prhs[3]);
  num_states = (int)dims[0];
  // rhythmic patterns
  dims = mxGetDimensions(prhs[4]);
  R = (int)dims[0];
  num_pos = (int)dims[1];
  // number of possible transitions
  dims = mxGetDimensions(prhs[1]);
  num_trans = (int)dims[0];
  // number of valid states
  dims = mxGetDimensions(prhs[7]);
  num_valid_states = (int)dims[0];
  
//associate outputs
  map_state_sequence = plhs[0] = mxCreateDoubleMatrix(num_frames,1,mxREAL);
//   debug_data = plhs[1] = mxCreateDoubleMatrix(num_states, 1, mxREAL);
  
//internal variables
//  mexPrintf("States=%d, Frames=%d\n", num_valid_states, num_frames);
//  mexPrintf("Size of Psi matrix: %.1f MB\n", num_valid_states*num_frames*4/1E6);
  psi = mxCreateNumericMatrix(num_valid_states, num_frames, mxINT32_CLASS, mxREAL);
  std::vector<double> delta(initial_prob_ptr, initial_prob_ptr+num_states);
  std::vector<double> prediction(initial_prob_ptr, initial_prob_ptr+num_states);
  
//associate pointers (get pointer to the first element of the real data)
  double *map_state_sequence_ptr = mxGetPr(map_state_sequence);
  psi_ptr = (mwSignedIndex*)mxGetData(psi);
// double *debug_data_ptr = mxGetPr(debug_data);
//start computing    
  for(i=0;i<num_frames;i++)
  {
      prev_end_state = -7;
      // loop over possible transitions
      for(i_trans=0;i_trans<num_trans;i_trans++)
      {
          current_start_state = (int)state_ids_i_ptr[i_trans]-1;
          current_end_state = (int)state_ids_j_ptr[i_trans]-1;
          if (current_end_state == prev_end_state) {
              temp = delta[current_start_state] * trans_prob_ij_ptr[i_trans];
             if ( temp > prediction[current_end_state]) { // found more probable precursor state
                 prediction[current_end_state] = temp;
                 idx = (int)(valid_states_ptr[current_end_state]-1)*num_frames+i;
                 psi_ptr[idx] = current_start_state;
              }         
          } else { // new state x(t)       
              prev_end_state = current_end_state;
              prediction[current_end_state] = delta[current_start_state] * trans_prob_ij_ptr[i_trans];   
              idx = (int)(valid_states_ptr[current_end_state]-1)*num_frames+i;
              psi_ptr[idx] = current_start_state;             
          }
      }
      sum_k = 0;
      for (i_state=0; i_state<num_states; i_state++) {
          if (valid_states_ptr[i_state] == 0) { // i_state is not a valid state (no transition)
              delta[i_state] = 0;
          }    
          else if (!mxIsNaN(state_2_r_pos_ptr[i_state])){
               // multiply with observation likelihood and sum up
              r = (int)state_2_r_pos_ptr[i_state]-1;
              p = (int)state_2_r_pos_ptr[i_state+num_states]-1;
              delta[i_state] = prediction[i_state] * obs_lik_ptr[r + p*R + i*R*num_pos]; 
              sum_k += delta[i_state];
          }
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
  map_state_sequence_ptr[num_frames-1] = (double)best_end_state+1;
  last_state_valid = (int)valid_states_ptr[best_end_state]-1;
  for (i=num_frames-1; i>0; i--) {
      idx = (int)(last_state_valid*num_frames+i);  // idx for psi
      map_state_sequence_ptr[i-1] = (double)psi_ptr[idx]+1;
      last_state_valid = valid_states_ptr[psi_ptr[idx]]-1;
  }
  mxDestroyArray(psi);
}