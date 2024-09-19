data = [
    0.9956 0.6221 0.6014;
    0.9956 0.6221 0.6014;
    0.9374 0.6330 0.6100;
    0.9911 0.6298 0.6222;
];

categories = {"SVM lineare","SVM lineare precomputato", "SVM non lineare chi 2", "SVM non lineare IK"};
% Create the grouped bar chart
groupLabels = {'Training Set','Validation set', 'Test set'};
figure;
b = bar(data);

% Set the x-axis labels
set(gca, 'XTickLabel', categories, 'XTickLabelRotation', 45);

% Add the legend
legend(groupLabels, 'Location', 'northeast');

% Add values on top of the bars
for k = 1:size(data, 2)
    text((1:size(data, 1)) - 0.25 + 0.25*(k-1), data(:, k), ...
        num2str(data(:, k), '%.4f'), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end

% Set labels
xlabel('Categories');
ylabel('Accuracy');
title('Grouped Bar Chart');
