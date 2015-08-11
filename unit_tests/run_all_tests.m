% This script runs all tests of the current folder
import matlab.unittest.TestSuite
% Create a suite from all test case files in the current folder
suiteFolder = TestSuite.fromFolder(pwd);
% Test it
result = run(suiteFolder);