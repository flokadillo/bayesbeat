vars = whos;
for k = 1:length(vars)
    eval(sprintf('c = %s',vars(k).name));
    samas = [c(1:2:end,:); zeros(1,size(c,2))];
    beats = [c(2:2:end,:); zeros(1,size(c,2))];
    cout = [samas(:); 0; beats(:)];
    cout = round(cout*100)/100;
    dlmwrite(vars(k).name, cout, 'precision','%.2f');
    clear beats samas c cout
end

function y = check(x)

end