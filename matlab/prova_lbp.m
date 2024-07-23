%% clear;
close all;
clc;
% DATASET
%dataset_dir='food'; %dataset_folder_name
dataset_dir='prova_resized'; %dataset_folder_name
%dataset_dir = '4_ObjectCategories';

% FEATURES extraction methods
% 'sift' for sparse features detection (SIFT descriptors computed at  
% Harris-Laplace keypoints) or 'dsift' for dense features detection (SIFT
% descriptors computed at a grid of overlapped patches

%desc_name = 'sift';
desc_name = 'dsift';
%desc_name = 'msdsift';

% FLAGS
do_feat_extraction = 0;
do_split_sets = 1;

do_form_codebook = 1;
do_feat_quantization = 1;

do_L2_NN_classification = 1;
do_chi2_NN_classification = 0;
do_svm_linar_classification = 1;
do_svm_llc_linar_classification = 0;
do_svm_precomp_linear_classification = 0;
do_svm_inter_classification = 0;
do_svm_chi2_classification = 0;

visualize_feat = 1;
visualize_words = 0;
visualize_confmat = 1;
visualize_res = 1;
%have_screen = ~isempty(getenv('DISPLAY'));
have_screen = 1;

% PATHS
basepath = '..';
wdir = pwd;
libsvmpath = [ wdir(1:end-6) fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

% BOW PARAMETERS
max_km_iters = 150; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

% number of images selected for training (e.g. 30 for Caltech-101)
num_train_img = 100; %numero per ogni classe
% number of images selected for test (e.g. 50 for Caltech-101)
num_test_img = 20;  %numero per ogni classe
% number of codewords (i.e. K for the k-means algorithm)
nwords_codebook = 500;
%NUmero massimo di immagini prendibili per ogni classe
num_max_img_per_classe = 140;

% image file extension
file_ext='jpg';

% Create a new dataset split
file_split = 'split.mat';
if do_split_sets    
    data = create_dataset_split_structure_from_unbalanced_sets(...
        fullfile(basepath, 'img', dataset_dir), ... 
        num_train_img, ...
        num_test_img , ...
        file_ext, ...
        num_max_img_per_classe); %numero di immagini massimo da considerare per classe
    save(fullfile(basepath,'img',dataset_dir,file_split),'data');
else
    load(fullfile(basepath,'img',dataset_dir,file_split));
end
classes = {data.classname}; % create cell array of class name strings

disp("Immagini caricate correttamente")

% Extract SIFT features fon training and test images
if do_feat_extraction   
    extract_sift_features(fullfile('..','img',dataset_dir),desc_name)    
    disp("Estrazione delle feature SIFT completata correttamente")
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 1: quantize pre-computed image features %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
info.base = basepath;
info.first = "img";
info.dsdir = dataset_dir;
info.desc_name = desc_name;
[trainLBP,testLBP] = lpb_extraction(data,length(classes),num_train_img,num_test_img,info);
disp("Parametri LBP estratti correttamente")
%% % Concatenate bof-histograms into training and test matrices 
for i = 1 : size(trainLBP,2)
    trainLBP(i).hist = double(trainLBP(i).hist);
end

for i = 1 : size(testLBP,2)
    testLBP(i).hist = double(testLBP(i).hist);
end

f_train=cat(1,trainLBP.hist);
f_test=cat(1,testLBP.hist);



% Construct label Concatenate bof-histograms into training and test matrices 
labels_train=cat(1,trainLBP.class);
labels_test=cat(1,testLBP.class);
%% 
if do_L2_NN_classification
    % Compute L2 distance between BOFs of test and training images
    f_l2dist=eucliddist(f_test,f_train);
    
    % Nearest neighbor classification (1-NN) using L2 distance
    [mv,mi] = min(f_l2dist,[],2); %val, colonna nel train
    f_l2lab = labels_train(mi);
    
    method_name='NN L2';
    acc=sum(f_l2lab==labels_test)/length(labels_test);
    fprintf('\n*** %s ***\nAccuracy = %1.4f%% (classification)\n',method_name,acc*100);
   
    % Compute classification accuracy
    compute_accuracy(data,labels_test, ...
        f_l2lab,classes, ...
        method_name,testLBP,...
        visualize_confmat & have_screen,... 
        visualize_res & have_screen);
end

%% 
if do_svm_linar_classification
    % cross-validation
    C_vals=log2space(7,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,f_train,opt_string);
    end
    %select the best C among C_vals and test your model on the testing set.
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,f_train,['-t 0 -c ' num2str(C_vals(ind))]);
    disp('*** SVM - linear ***');
    svm_lab=svmpredict(labels_test,f_test,model);
    
    method_name='SVM linear';
    % Compute classification accuracy
    compute_accuracy_lbp(data,labels_test,svm_lab,classes,method_name,testLBP,...
                      1,... 
                      1);
end
