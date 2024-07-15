function [ ] = visualize_results( classes, desc_test, labels_test, labels_result )

%VISUALIZE_EXAMPLES Illustrate correcly classified and missclassified 
%samples of each class.

figure;

for i=1:length(classes)
    
    ind=find(labels_test==i);
    %labels_result %valori predetti per j-esima immagine della categoria i
    %labels_test
    %bof_chi2acc_class=mean(result_labels(ind)==labels_test(ind));
    indcorr=ind(find(labels_result(ind)==labels_test(ind)));
    indmiss=ind(find(labels_result(ind)~=labels_test(ind)));
    
    clf
    imgcorr={};
    if length(indcorr)
        for j=1:length(indcorr) 
            imgcorr{end+1}=imread(desc_test(indcorr(j)).imgfname);
        end
        subplot(1,2,1), showimage(combimage(imgcorr,[],1))
        title(sprintf('%d Correctly classified %s images',length(indcorr),classes{i}))
    end
    
    imgmiss={};
    if length(indmiss)
        % for j=1:length(indmiss)
        %     imgmiss{end+1}=imread(desc_test(indmiss(j)).imgfname);
        % end
        % subplot(1,2,2), showimage(combimage(imgmiss,[],1))
        % title(sprintf('%d Miss-classified %s images',length(indmiss),classes{i}))

        for j=1:length(indmiss)
            imgmiss{end+1} = imread(desc_test(indmiss(j)).imgfname);
        end

        max_images_per_row = 6; % Numero massimo di immagini per riga
        
        figure; % Crea una nuova figura
        
        % Numero di immagini da visualizzare
        numImages = length(imgmiss);

        % Calcola il numero di righe necessario
        numRows = ceil(numImages / max_images_per_row);

        for k = 1:numImages
            % Calcola la posizione del subplot
            row = ceil(k / max_images_per_row);
            col = mod(k-1, max_images_per_row) + 1;
            subplot(numRows, max_images_per_row, (row-1) * max_images_per_row + col);
            imshow(imgmiss{k}); % Visualizza l'immagine


            % Aggiungi un sottotitolo sotto ciascuna immagine
            classe_scorretta_predetta = classes(labels_result(indmiss(k)));
            
            classe_scorretta_predetta = classe_scorretta_predetta{1};
            
            subtitle = sprintf('%s', classe_scorretta_predetta);
            title(subtitle, 'FontSize', 10, 'FontWeight', 'normal');
        end
        
        % Aggiungi un titolo generale alla figura
        sgtitle(sprintf('%d Miss-classified %s images', length(indmiss), classes{i}));

    end
    
    pause;
end

end

