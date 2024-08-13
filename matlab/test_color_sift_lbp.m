
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ICPR 2014 Tutorial                                                     %
%  Hands on Advanced Bag-of-Words Models for Visual Recognition           %
%                                                                         %
%  Instructors:                                                           %
%  L. Ballan     <lamberto.ballan@unifi.it>                               %
%  L. Seidenari  <lorenzo.seidenari@unifi.it>                             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%FILIPPO TI STO ODIANDO 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   BOW pipeline: Image classification using bag-of-features              %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   Part 1:  Load and quantize pre-computed image features                %
%   Part 2:  Represent images by histograms of quantized features         %
%   Part 3:  Classify images with nearest neighbor classifier             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%run('path_to_vlfeat/vlfeat-0.9.21/toolbox/vl_setup');

clear;
close all;
clc;
% DATASET
%dataset_dir='4_ObjectCategories';
%dataset_dir = '15_ObjectCategories';
dataset_dir = 'ds';
%dataset_dir = 'prova_resized_bn_2';

% FEATURES extraction methods
% 'sift' for sparse features detection (SIFT descriptors computed at
% Harris-Laplace keypoints) or 'dsift' for dense features detection (SIFT
% descriptors computed at a grid of overlapped patches

desc_name = 'color_sift';


% FLAGS
do_feat_extraction = 1;
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
have_screen = 0;

% PATHS
basepath = '..';
wdir = pwd;
libsvmpath = [ wdir(1:end-6) fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

% BOW PARAMETERS
max_km_iters = 1500; % maximum number of iterations for k-means
nfeat_codebook = 280000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

% number of images selected for training (e.g. 30 for Caltech-101)
num_train_img = 150; %numero per ogni classe
%number of images selected for validation
num_val_img = 20;
% number of images selected for test (e.g. 50 for Caltech-101)
num_test_img = 30;  %numero per ogni classe
% number of codewords (i.e. K for the k-means algorithm)
nwords_codebook = 1200;
%NUmero massimo di immagini prendibili per ogni classe
num_max_img_per_classe = 200;

% image file extension
file_ext='jpg';

%% 
% Create a new dataset split
file_split = 'split.mat';
if do_split_sets
    data = create_dataset_split_structure_from_unbalanced_sets_val(fullfile(basepath, 'img', ...
        dataset_dir),num_train_img,num_val_img,num_test_img ,file_ext,num_max_img_per_classe);
    save(fullfile(basepath,'img',dataset_dir,file_split),'data');
else
    load(fullfile(basepath,'img',dataset_dir,file_split));
end
classes = {data.classname}; %
% create cell array of class name strings

% Extract SIFT features fon training and test images
if do_feat_extraction
    extract_sift_features(fullfile('..','img',dataset_dir),desc_name)
end



%passa le immagini una ad una alla funzuone per pyramid

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 1: quantize pre-computed image features %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Load pre-computed SIFT features for training images

% The resulting structure array 'desc' will contain one
% entry per images with the following fields:
%  desc(i).r :    Nx1 array with y-coordinates for N SIFT features
%  desc(i).c :    Nx1 array with x-coordinates for N SIFT features
%  desc(i).rad :  Nx1 array with radius for N SIFT features
%  desc(i).sift : Nx128 array with N SIFT descriptors
%  desc(i).imgfname : file name of original image
clear desc_train;
lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files_val(data,i,file_ext,desc_name,'train');
    for j = 1:length(images_descs)
         fprintf('Loading %d/%d \n',j,length(images_descs));
        for l = 1 : 3
            fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-11),'_',num2str(l),'.color_sift']);
            tmp = load(fname,'-mat');
            tmp.desc.class=i;
            tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
            desc_train(lasti,l)=tmp.desc;  
            desc_train(lasti,l).sift = single(desc_train(lasti,l).sift);
        
        end
        lasti=lasti+1;
    end
end


%% Visualize SIFT features for training images
if (visualize_feat && have_screen)
    nti=2;
    fprintf('\nVisualize features for %d training images\n', nti);
    imgind=randperm(length(desc_train));
    for i=1:nti
        d=desc_train(imgind(i));
        pattern = '_\d';
        replace = '';
        new_name = regexprep(d.imgfname,pattern,replace);
        clf, showimage(imread(strrep(new_name,'_train','')));
        x=d.c;
        y=d.r;
        rad=d.rad;
        showcirclefeaturesrad([x,y,rad]);
        title(sprintf('%d features in %s',length(d.c),d.imgfname));
        %pause
    end
end


%% Load pre-computed SIFT features for
lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files_val(data,i,file_ext,desc_name,'test');
    for j = 1:length(images_descs)
        fprintf('Loading %d/%d \n',j,length(images_descs));
        for l = 1 : 3
            fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-11),'_',num2str(l),'.color_sift']);
            
            tmp = load(fname,'-mat');
            tmp.desc.class=i;
            tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
            desc_test(lasti,l)=tmp.desc;
            desc_test(lasti,l).sift = single(desc_test(lasti,l).sift);
        end
        lasti=lasti+1;
    end;
end;



%% Load pre-computed SIFT features for validation images

lasti=1;
for i = 1:length(data)
    images_descs = get_descriptors_files_val(data,i,file_ext,desc_name,'val');

    for j = 1:length(images_descs)
         fprintf('Loading %d/%d \n',j,length(images_descs));
        for l = 1 : 3
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,[images_descs{j}(1:end-11),'_',num2str(l),'.color_sift']);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_val(lasti,l)=tmp.desc;
        desc_val(lasti,l).sift = single(desc_val(lasti,l).sift);
        end
        lasti=lasti+1;
    end;
end;


%% Build visual vocabulary using k-means %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_form_codebook
    fprintf('\nBuild visual vocabulary:\n');

    % concatenate all descriptors from all images into a n x d matrix
    DESC = [];
    labels_train_1 = cat(1,desc_train(:,1).class);
    labels_train_2 = cat(1,desc_train(:,2).class);
    labels_train_3 = cat(1,desc_train(:,3).class);
    for i=1:length(data)
     fprintf('Processing %d/%d \n',i,length(data));
     


        desc_train_1 = desc_train(:,1);
        desc_train_2 = desc_train(:,2);
        desc_train_3 = desc_train(:,3);
        
        desc_class_1 = desc_train_1(labels_train_1==i);
        desc_class_2 = desc_train_2(labels_train_2==i); 
        desc_class_3 = desc_train_3(labels_train_3==i); 
        
        randimages = randperm(num_train_img);
        randimages =randimages(1:5);
        DESC = vertcat(DESC,desc_class_1(randimages,1).sift,desc_class_2(randimages,1).sift,desc_class_3(randimages,1).sift);


    end
%% 

    % sample random M (e.g. M=20,000) descriptors from all training descriptors
    r = randperm(size(DESC,1));
    r = r(1:min(length(r),nfeat_codebook));

    DESC = DESC(r,:);
    % run k-means
    K = nwords_codebook; % size of visual vocabulary
    fprintf('running k-means clustering of %d points into %d clusters...\n',...
        size(DESC,1),K)
    % input matrix needs to be transposed as the k-means function expects
    % one point per column rather than per row

    % form options structure for clustering
    cluster_options.maxiters = max_km_iters;
    cluster_options.verbose  = 1;

    [VC] = kmeans_bo(double(DESC),K,max_km_iters);%visual codebook
    VC = VC';%transpose for compatibility with following functions
    clear DESC;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 1: K-means Descriptor quantization                           %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% K-means descriptor quantization means assignment of each feature
% descriptor with the identity of its nearest cluster mean, i.e.
% visual word. Your task is to quantize SIFT descriptors in all
% training and test images using the visual dictionary 'VC'
% constructed above.
%
% TODO:
% 1.1 compute Euclidean distances between VC and all descriptors
%     in each training and test image. Hint: to compute all-to-all
%     distance matrix for two sets of descriptors D1 & D2 use
%     dmat=eucliddist(D1,D2);
% 1.2 compute visual word ID for each feature by minimizing
%     the distance between feature SIFT descriptors and VC.
%     Hint: apply 'min' function to 'dmat' computed above along
%     the dimension (1 or 2) corresponding to VC, i.g.:
%     [mv,visword]=min(dmat,[],2); if you compute dmat as
%     dmat=eucliddist(dscr(i).sift,VC);

if do_feat_quantization
    fprintf('\nFeature quantization (hard-assignment)...\n');
    for i=1:length(desc_train)
        fprintf('Feature quantization training set: %d/%d \n',i,length(desc_train));
        for j = 1 : 3
            sift = desc_train(i,j).sift(:,:);
            dmat = eucliddist(sift,VC);
            [quantdist,visword] = min(dmat,[],2);
            % save feature labels
            desc_train(i,j).visword = visword;
            desc_train(i,j).quantdist = quantdist;
        end
    end

    for i=1:length(desc_test)
         fprintf('Feature quantization test set: %d/%d \n',i,length(desc_test));
         for j = 1 : 3
            sift = desc_test(i,j).sift(:,:);
            dmat = eucliddist(sift,VC);
            [quantdist,visword] = min(dmat,[],2);
            % save feature labels
            desc_test(i,j).visword = visword;
            desc_test(i,j).quantdist = quantdist;
         end
    end

    
    for i=1:length(desc_val)
        fprintf('Feature quantization validation set: %d/%d \n',i,length(desc_val));
         for j = 1 : 3
            sift = desc_val(i,j).sift(:,:);
            dmat = eucliddist(sift,VC);
            [quantdist,visword] = min(dmat,[],2);
            % save feature labels
            desc_val(i,j).visword = visword;
            desc_val(i,j).quantdist = quantdist;
         end
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 1                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Visualize visual words (i.e. clusters) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  To visually verify feature quantization computed above, we can show
%  image patches corresponding to the same visual word.

if (visualize_words && have_screen)
    figure;
    %num_words = size(VC,1) % loop over all visual word types
    num_words = 10;
    fprintf('\nVisualize visual words (%d examples)\n', num_words);
    for i=1:num_words
        patches={};
        for j=1:length(desc_train) % loop over all images
            d=desc_train(j);
            ind=find(d.visword==i);
            if length(ind)
                %img=imread(strrep(d.imgfname,'_train',''));
                d=desc_train(imgind(i));
                pattern = '_\d';
                replace = '';
                new_name = regexprep(d.imgfname,pattern,replace);
                img=im2gray(imread(new_name));

                x=d.c(ind); y=d.r(ind); r=d.rad(ind);
                bbox=[x-2*r y-2*r x+2*r y+2*r];
                for k=1:length(ind) % collect patches of a visual word i in image j
                    patches{end+1}=cropbbox(img,bbox(k,:));
                end
            end
        end
        % display all patches of the visual word i
        clf, showimage(combimage(patches,[],1.5))
        title(sprintf('%d examples of Visual Word #%d',length(patches),i))
        %pause
    end
end


%CACCA%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 2: represent images with BOF histograms %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 2: Bag-of-Features image classification                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Represent each image by the normalized histogram of visual
% word labels of its features. Compute word histogram H over
% the whole image, normalize histograms w.r.t. L1-norm.
%
% TODO:
% 2.1 for each training and test image compute H. Hint: use
%     Matlab function 'histc' to compute histograms.

fprintf('\n%%%%%%%%%%  Start BoF   %%%%%%%%%%%%%%%');

N = size(VC,1); % number of visual words

for i=1:length(desc_train)
    for j = 1 : 3 
        visword = desc_train(i,j).visword;
    
        H = histc(visword,[1:nwords_codebook]);
    
        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end
    
        % save histograms
        desc_train(i,j).bof=H(:)';
    end
end

for i=1:length(desc_test)
    for j = 1 : 3 
        visword = desc_test(i,j).visword;
        H = histc(visword,[1:nwords_codebook]);
    
        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end
    
        % save histograms
        desc_test(i,j).bof=H(:)';
    end
end

for i=1:length(desc_val)
    for j = 1 : 3 
        visword = desc_val(i,j).visword;
        H = histc(visword,[1:nwords_codebook]);
    
        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end
    
        % save histograms
        desc_val(i,j).bof=H(:)';
    end
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 3: image classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n%%%%%%%%%%  Manipulatinf BoFs   %%%%%%%%%%%%%%%');
desc_train_bof = [];
for i = 1 : size(desc_train)
    desc_train_bof(end+1,:) =horzcat(desc_train(i,1).bof,desc_train(i,2).bof,desc_train(i,3).bof);
end


desc_val_bof = [];
for i = 1 : size(desc_val)
    desc_val_bof(end+1,:) =horzcat(desc_val(i,1).bof,desc_val(i,2).bof,desc_val(i,3).bof);
end


desc_test_bof = [];
for i = 1 : size(desc_test)
    desc_test_bof(end+1,:) =horzcat(desc_test(i,1).bof,desc_test(i,2).bof,desc_test(i,3).bof);
end


% Concatenate bof-histograms into training and test matrices

%%ultmo scritto
bof_train=cat(1,desc_train.bof);
bof_test=cat(1,desc_test.bof);
if do_svm_llc_linar_classification
    llc_train = cat(1,desc_train.llc);
    llc_test = cat(1,desc_test.llc);
end
%% %% 
%%Compute  LBP for each image

%definition about path information
info.base = basepath;
info.first = "img";
info.dsdir = dataset_dir;
info.desc_name = desc_name;
[trainLBP,testLBP] = lbp_color_extraction(data,length(classes),num_train_img,num_test_img,info);
for i = 1 : size(trainLBP,2)
    trainLBP(i).hist = double(trainLBP(i).hist);
end

for i = 1 : size(testLBP,2)
    testLBP(i).hist = double(testLBP(i).hist);
end

%% 
f_train=cat(1,trainLBP.hist);
f_test=cat(1,testLBP.hist);
%% 

new_bof_train = [bof_train,f_train];
new_bof_test = [bof_test,f_test];

%% 
% Construct label Concatenate bof-histograms into training and test matrices 
labels_train=cat(1,desc_train.class);
labels_test=cat(1,desc_test.class);


%% 4.3 & 4.4: CHI-2 KERNEL (pre-compute kernel) %%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_svm_precomp_linear_classification
    % compute kernel matrix
    Ktrain = new_bof_train*new_bof_train';
    Ktest = new_bof_test*new_bof_train';

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
    disp('*** SVM - precomputed linear kernel ***');
    precomp_svm_lab=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    
    method_name='SVM precomp linear';
    % Compute classification accuracy
    compute_accuracy(data,labels_test,precomp_svm_lab,classes,method_name,desc_test,...
                      visualize_confmat & have_screen,... 
                      visualize_res & have_screen);
    % result is the same??? must be!
end




%display accurancy bars
%methods_name(end) = [];
%f_accurancy_final= figure;
%f_accurancy_final.Name = "Accurancy methods";
%display_bar_accurancy(f_accurancy_final, bar_values,methods_name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 4.3 and 4.4                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
if do_svm_linar_classification
    % cross-validation
    C_vals=log2space(7,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,new_bof_train,opt_string);
    end
    %select the best C among C_vals and test your model on the testing set.
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,new_bof_train,['-t 0 -c ' num2str(C_vals(ind))]);
    disp('*** SVM - linear ***');
    svm_lab=svmpredict(labels_test,new_bof_test,model);
    
    method_name='SVM linear';
    % Compute classification accuracy
    compute_accuracy(data,labels_test,svm_lab,classes,method_name,desc_test,...
                      visualize_confmat & have_screen,... 
                      visualize_res & have_screen);
end

%% 
if do_svm_chi2_classification    
    % compute kernel matrix
    Ktrain = kernel_expchi2(new_bof_train,new_bof_train);
    Ktest = kernel_expchi2(new_bof_test,new_bof_train);
    
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
    [precomp_chi2_svm_lab,conf]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    
    method_name='SVM Chi2';
    % Compute classification accuracy
    compute_accuracy(data,labels_test,precomp_chi2_svm_lab,classes,method_name,desc_test,...
                      visualize_confmat & have_screen,... 
                      visualize_res & have_screen);
end
