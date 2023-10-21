%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Yassine Kebbati
% Date: 20/12/2019
% Control NN-PID-Autonomous_Driving
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testDir = 'D:\MATLAB\mat\PhD_Clean_Code\NN_PID'
addpath(testDir)

%Specify Python Executable Library.
pcPythonExe = 'C:\Users\yassine\anaconda3\envs\tf_PID\python';
[ver, exec, loaded]	= pyversion(pcPythonExe); pyversion

% Ensure python-matlab integration code is on matlab path.
pyFolder = fullfile(matlabroot, 'toolbox', 'matlab', 'external', 'interfaces', 'python');
addpath(pyFolder);

% Folder containing all relevant python libraries.
pyLibraryFolder = 'C:\Users\yassine\anaconda3\envs\tf_PID\Lib\site-packages';
% Add folders to python system path.
insert(py.sys.path, int64(0), testDir);
insert(py.sys.path, int64(0), pyFolder);
insert(py.sys.path, int64(0), pyLibraryFolder);

%% Call python script.
py_test_mod = py.importlib.import_module('test')
% % Using system call instead of matlab-python integration functionality.
% [result, status] = python('test_from_import.py') % Does not return error.