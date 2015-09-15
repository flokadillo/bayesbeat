This file explains how to perform unit testing with the bayesbeat package. First make sure that the Bayesbeat folder is in the search path. If not, add it using <addpath>.

# Run all tests
import matlab.unittest.TestSuite
suiteFolder = TestSuite.fromFolder(pwd);
result = run(suiteFolder);

# Run all tests of a test class
run(ExamplesTest);

# Run a specific test of a test class
run(ExamplesTest, 'testEx3');
