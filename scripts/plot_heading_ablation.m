%% Re-export the verified Heading Prediction results in the paper style.
scriptDir=fileparts(mfilename('fullpath'));
addpath(scriptDir);
render_paper_results('heading');
