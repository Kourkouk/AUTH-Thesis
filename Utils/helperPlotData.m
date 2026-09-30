function  helperPlotData(DS,Indices,net1,net2)
% This function is only intended to support this example. It may be changed
% or removed in a future release. 
arguments
    DS 
    Indices
    net1 =[]
    net2 = []
end
fs = 200;
N = numel(Indices);
M = 2;
if ~isempty(net1)
    M = M + 1;
end
if ~isempty(net2)
    M = M + 1;
end

tiledlayout(M, N, 'Padding', 'none', 'TileSpacing', 'compact');
for i = 1:N
    idx = Indices(i);
    ds = subset(DS,idx);
    [~,name,~] = fileparts(ds.UnderlyingDatastores{1}.Files{1});
    data = read(ds);
    radar = data{1};
    ecg = data{2};
    t = linspace(0,length(radar)/fs,length(radar));

    nexttile(i)
    plot(t,radar)
    title(["Sample",regexprep(name, {'_','radar'}, '')])
    xlabel(["Radar Signal","Time (s)"])
    grid on

    nexttile(N+i)
    plot(t,ecg)
    xlabel(["Measured ECG Signal","Time (s)"])
    ylim([-0.3,1])
    grid on

    if ~isempty(net1)
        nexttile(2*N+i)
        y = minibatchpredict(net1,radar,InputDataFormats="CTB");
        plot(t,y)
        grid on
        ylim([-0.3,1])
        xlabel(["Reconstructed ECG Signal","Time (s)"])
    end

    if ~isempty(net2)
        nexttile(3*N+i)
        y = minibatchpredict(net2,radar,InputDataFormats="CTB");
        hold on
        plot(t,y)
        hold off
        grid on
        ylim([-0.3,1])
        xlabel(["Reconstructed ECG Signal", "with modwtLayer","Time (s)"])
    end

end
set(gcf,'Position',[0 0 300*N,150*M])
end