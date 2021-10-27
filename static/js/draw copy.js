function draw_3_scenarios(ctx,dates,real,best,normal,worst) {
    const skipped = (ctx, value) => ctx.p0.skip || ctx.p1.skip ? value : undefined;
    const down = (ctx, value) => ctx.p0.parsed.y > ctx.p1.parsed.y ? value : undefined;

    const genericOptions = {
        fill: false,
        interaction: {
            intersect: false
        },
        radius: 0,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero: true
                }
            }]
        },
        elements: {
            point:{
                radius: 0
            }
        }
    };

    const config = {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Real',
                data: real,
                borderColor: 'rgb(41, 128, 185)',
            },
            {
                label: 'Best Case',
                data: best,
                borderDash: [3,3],
                borderColor: 'rgb(46, 204, 113)',
            },
            {
                label: 'Normal Case',
                data: normal,
                borderDash: [3, 3],
                borderColor: 'rgb(243, 156, 18)',
            },
            {
                label: 'Worst Case',
                data: worst,
                borderDash: [3, 3],
                borderColor: 'rgb(231, 76, 60)',
            },
            ]
        },
        options: genericOptions
    };

    // Any of the following formats may be used
    // var ctx = document.getElementById('myChart');
    // var ctx = document.getElementById('myChart').getContext('2d');
    var myChart = new Chart(ctx, config)
}
