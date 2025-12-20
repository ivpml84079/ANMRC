function [IR] = predict_IR(rad_diff, scale_ratio)

first_filter = histogram_filter(rad_diff, 0.1, "angel", []);

scale_ratio = scale_ratio(first_filter);
second_filter = histogram_filter(scale_ratio, 0.5, "length", first_filter);

IR  = size(second_filter, 1) / size(rad_diff, 1);