%% clear;
clear;
close all;
clc;
% DATASET
%dataset_dir='food'; %dataset_folder_name
dataset_dir='ds2'; %dataset_folder_name
%dataset_dir = 'prova_resized_bn_2';
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

do_L2_NN_classification = 0;
do_chi2_NN_classification = 0;
do_svm_linar_classification = 1;
do_svm_llc_linar_classification = 0;
do_svm_precomp_linear_classification = 1;
do_svm_inter_classification = 1;
do_svm_chi2_classification = 1;

visualize_feat = 0;
visualize_words = 0;
visualize_confmat = 0;
visualize_res = 0;
%have_screen = ~isempty(getenv('DISPLAY'));
have_screen = 0;

% PATHS
basepath = '..';
wdir = pwd;
libsvmpath = [ wdir(1:end-6) fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

% BOW PARAMETERS
max_km_iters = 1500; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

% number of images selected for training (e.g. 30 for Caltech-101)
% number of images selected for training (e.g. 30 for Caltech-101)
num_train_img = 142; %numero per ogni classe

%number of images selected fo validation
num_val_img = 48;
% number of images selected for test (e.g. 50 for Caltech-101)
num_test_img = 48;  %numero per ogni classe
% number of codewords (i.e. K for the k-means algorithm)
nwords_codebook = 1200;
%NUmero massimo di immagini prendibili per ogni classe
num_max_img_per_classe = 238;

% image file extension
file_ext='jpg';

% Create a new dataset split
file_split = 'split.mat';
if do_split_sets    
    data = create_dataset_split_structure_from_unbalanced_sets_val(...
        fullfile(basepath, 'img', dataset_dir), ... 
        num_train_img, ...
        num_val_img,...
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
[trainLBP,valLBP,testLBP] = lbp_val_extraction(data,length(classes),num_train_img,num_val_img,num_test_img,info);
disp("Parametri LBP estratti correttamente")
%% % Concatenate bof-histograms into training and test matrices 
for i = 1 : size(trainLBP,2)
    trainLBP(i).hist = double(trainLBP(i).hist);
end

for i = 1 : size(valLBP,2)
    valLBP(i).hist = double(valLBP(i).hist);
end

for i = 1 : size(testLBP,2)
    testLBP(i).hist = double(testLBP(i).hist);
end

f_train=cat(1,trainLBP.hist);
f_val = cat(1,valLBP.hist);
f_test=cat(1,testLBP.hist);



% Construct label Concatenate bof-histograms into training and test matrices 
labels_train=cat(1,trainLBP.class);
labels_val = cat(1,valLBP.class);
labels_test=cat(1,testLBP.class);
%% 
if 0%do_L2_NN_classification
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
if 0%do_svm_linar_classification
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
    disp('*** SVM - linear (test) ***');
    svm_lab_test=svmpredict(labels_test,f_test,model);
    method_name='SVM linear';
    compute_accuracy_lbp(data,labels_test,svm_lab_test,classes,method_name,testLBP, visualize_confmat, visualize_res);

    disp('*** SVM - linear (val) ***');
    svm_lab_val=svmpredict(labels_val,f_val,model);
    method_name='SVM linear';
    compute_accuracy_lbp(data,labels_val,svm_lab_val,classes,method_name,valLBP, visualize_confmat, visualize_res);
    
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 4: Image classification: SVM classifier                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pre-computed LINEAR KERNELS. 
% Repeat linear SVM image classification; let's try this with a 
% pre-computed kernel.
%
% TODO:
% 4.1 Compute the kernel matrix (i.e. a matrix of scalar products) and
%     use the LIBSVM precomputed kernel interface.
%     This should produce the same results.


if 0%do_svm_precomp_linear_classification
    % compute kernel matrix
    Ktrain = f_train*f_train';
    Ktest = f_test*f_train';
    Kval = f_val*f_train';

    % cross-validation
    C_vals=log2space(7,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);
    
    % train the model and test
    model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))]);
    % we supply the missing scalar product (actually the values of 
    % non-support vectors could be left as zeros.... 
    % consider this if the kernel is computationally inefficient.
    disp('*** SVM - precomputed linear kernel (test) ***');
    precomp_svm_lab_test=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    method_name='SVM precomp linear';
    compute_accuracy_lbp(data,labels_test,precomp_svm_lab_test,classes,method_name,testLBP, visualize_confmat, visualize_res);

    % result is the same??? must be!
    disp('*** SVM - precomputed linear kernel (val) ***');
    precomp_svm_lab_val=svmpredict(labels_val,[(1:size(Kval,1))' Kval],model);
    method_name='SVM precomp linear';
    compute_accuracy_lbp(data,labels_val,precomp_svm_lab_val,classes,method_name,valLBP, visualize_confmat, visualize_res);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 4.1                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Pre-computed NON-LINAR KERNELS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TODO:
% 4.2 Train the SVM with a precomputed non-linear histogram intersection 
%     kernel and select the best C parameter for the trained model using  
%     cross-validation.
% 4.3 Experiment with other different non-linear kernels: RBF and Chi^2.
%     Chi^2 must be precomputed as in the previous exercise.
% 4.4 Certain kernels have other parameters (e.g. gamma for RBF/Chi^2)... 
%     implement a cross-validation procedure to select the optimal 
%     parameters (as in 3).


%% 4.2: INTERSECTION KERNEL (pre-compute kernel) %%%%%%%%%%%%%%%%%%%%%%%%%%
% try a non-linear svm with the histogram intersection kernel!

if 0%do_svm_inter_classification
    Ktrain=zeros(size(f_train,1),size(f_train,1));
    for i=1:size(f_train,1)
        for j=1:size(f_train,1)
            hists = [f_train(i,:);f_train(j,:)];
            Ktrain(i,j)=sum(min(hists));
        end
    end

    Ktest=zeros(size(f_test,1),size(f_train,1));
    for i=1:size(f_test,1)
        for j=1:size(f_train,1)
            hists = [f_test(i,:);f_train(j,:)];
            Ktest(i,j)=sum(min(hists));
        end
    end

    Kval=zeros(size(f_val,1),size(f_train,1));
    for i=1:size(f_val,1)
        for j=1:size(f_train,1)
            hists = [f_val(i,:);f_train(j,:)];
            Kval(i,j)=sum(min(hists));
        end
    end

    % cross-validation
    C_vals=log2space(3,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
    % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros.... consider this if the kernel is computationally inefficient.
    disp('*** SVM - intersection kernel (test) ***');
    [precomp_ik_svm_lab_test,conf_test]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    method_name='SVM IK';
    compute_accuracy(data,labels_test,precomp_ik_svm_lab,classes,method_name,testLBP,0,0);

    disp('*** SVM - intersection kernel (val) ***');
    [precomp_ik_svm_lab_val,conf_val]=svmpredict(labels_val,[(1:size(Kval,1))' Kval],model);
    method_name='SVM IK';
    compute_accuracy(data,labels_val,precomp_ik_svm_lab_val,classes,method_name,valLBP,0,0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 4.2                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 4.3 & 4.4: CHI-2 KERNEL (pre-compute kernel) %%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_svm_chi2_classification

    fprintf('\n%%%%%%%%%% COMPUTE KERNELS  %%%%%%%%%%%%%%%');

    
    % compute kernel matrix
    Ktrain = kernel_expchi2(f_train,f_train);
    Ktest = kernel_expchi2(f_test,f_train);
    Kval = kernel_expchi2(f_val,f_train);
    

   
    % Compute the chi-squared kernel matrix
    
    fprintf('\n%%%%%%%%%% TRAINING THE MODEL  %%%%%%%%%%%%%%%');
    % cross-validation
    C_vals=log2space(2,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
    % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros....
    % consider this if the kernel is computationally inefficient.
    disp('*** SVM - Chi2 kernel ***');
    [precomp_chi2_svm_lab_test,conf]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    [precomp_chi2_svm_lab_val,conf]=svmpredict(labels_val,[(1:size(Kval,1))' Kval],model);
    [precomp_chi2_svm_lab_train,conf]=svmpredict(labels_train,[(1:size(Ktrain,1))' Ktrain],model);

    method_name="SVM Chi2";
    compute_accuracy_lbp(data,labels_train,precomp_chi2_svm_lab_train,classes,method_name,trainLBP,...
        visualize_confmat & have_screen,...
        visualize_res & have_screen);
    compute_accuracy_lbp(data,labels_val,precomp_chi2_svm_lab_val,classes,method_name,valLBP,...
        1,...
        1);
    compute_accuracy_lbp(data,labels_test,precomp_chi2_svm_lab_test,classes,method_name,testLBP,...
        0,...
        0);
    %methods_name(end+1) = method_name + ' k=' + nwords_codebook;
    %bar_values(end+1, :) = [acc_SVM_CHI2_train,acc_SVM_CHI2_val,acc_SVM_CHI2_test];

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 4.3 and 4.4                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
