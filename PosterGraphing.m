% Quick program just to plot averaged dose-response curves with error bars.

function PosterGraphing()

FullTable = readmatrix("brlA_Data_Master.xlsx",'Sheet','Dose-ResponseAvgs');
MicaXVals = [0; 5; 10; 15; 20]; % ng/mL

figure;

title('Dynamic brlA Dose-Response Data')

subplot(2,3,1)
errorbar(MicaXVals,FullTable(2:6,2)',FullTable(9:13,2)')
xlabel('Micafungin Concentration (ng/mL)')
ylabel('Fold Change, brlA')
title('Dose-Response: 0 minutes')
xlim([0 20.5])

subplot(2,3,2)
errorbar(MicaXVals,FullTable(2:6,3)',FullTable(9:13,3)')
xlabel('Micafungin Concentration (ng/mL)')
ylabel('Fold Change, brlA')
title('Dose-Response: 10 minutes')
xlim([0 20.5])

subplot(2,3,3)
errorbar(MicaXVals,FullTable(2:6,4)',FullTable(9:13,4)')
xlabel('Micafungin Concentration (ng/mL)')
ylabel('Fold Change, brlA')
title('Dose-Response: 20 minutes')
xlim([0 20.5])

subplot(2,3,4)
errorbar(MicaXVals,FullTable(2:6,5)',FullTable(9:13,5)')
xlabel('Micafungin Concentration (ng/mL)')
ylabel('Fold Change, brlA')
title('Dose-Response: 30 minutes')
xlim([0 20.5])

subplot(2,3,5)
errorbar(MicaXVals,FullTable(2:6,6)',FullTable(9:13,6)')
xlabel('Micafungin Concentration (ng/mL)')
ylabel('Fold Change, brlA')
title('Dose-Response: 60 minutes')
xlim([0 20.5])

subplot(2,3,6)
errorbar(MicaXVals,FullTable(2:6,7)',FullTable(9:13,7)')
xlabel('Micafungin Concentration (ng/mL)')
ylabel('Fold Change, brlA')
title('Dose-Response: 90 minutes')
xlim([0 20.5])

end