clc
% This script runs all tests of the current folder
import matlab.unittest.TestSuite
% Create a suite from all test case files in the current folder
suiteFolder = TestSuite.fromFolder(pwd);
% Add the bayesbeat class to the search path
[func_path, ~, ~] = fileparts(mfilename('fullpath'));
addpath(fullfile(func_path, '../src'))
% Test it
result = run(suiteFolder);