function draw_3_scenarios(id,dates,real,best,normal,worst) {
    var ctx = document.getElementById(id);
    const skipped = (ctx, value) => ctx.p0.skip || ctx.p1.skip ? value : undefined;
    const down = (ctx, value) => ctx.p0.parsed.y > ctx.p1.parsed.y ? value : undefined;
    
    for (let i = 0; i < real.length; i++) {
        best[i] = NaN
        normal[i] = NaN
        worst[i] = NaN
      }
    best[real.length-1] = real[real.length-1]
    normal[real.length-1] = real[real.length-1]
    worst[real.length-1] = real[real.length-1]

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
                label: 'Actual',
                data: real,
                borderColor: 'rgb(41, 128, 185)',
            },
            {
                label: 'Best Scenario',
                data: best,
                // borderDash: [3,3],
                borderColor: 'rgb(46, 204, 113)',
            },
            {
                label: 'Normal Scenario',
                data: normal,
                // borderDash: [3, 3],
                borderColor: 'rgb(243, 156, 18)',
            },
            {
                label: 'Worst Scenario',
                data: worst,
                // borderDash: [3, 3],
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


function draw_chart(id, data, days)
   {
        dates = data['dates']
        actual_data = data['actual']
        scenario = data['scenario']
        best_scenario = data['data']
        days = Number(days)
        days = days==-1 ? dates.length : days + 31
        return new Chart(id, {
            type: 'bar',
            data: {
                labels: dates.slice(-days),
                datasets: [
                {
                    label: 'Actual',
                    data: actual_data.slice(-days),
                    borderWidth: 1.,
                    barPercentage: 1.0,
                    categoryPercentage: 1.0,
                    backgroundColor: "#caf270",
                    stack: 'Stack 0',
                },
                {
                    label: scenario,
                    data: best_scenario.slice(-days),
                    barPercentage: 1.0,
                    categoryPercentage: 1.0,
                    backgroundColor: "#2e5468",
                    stack: 'Stack 0',
                }
                ]
            },
            options: {
                // tooltips: {
                //     displayColors: true,
                //     callbacks:{
                //         mode: 'x',
                //     },
                // },
                scales: {
                    xAxes: [{
                        stacked: true,
                        gridLines: {
                            display: false,
                        },
                    }],
                    yAxes: [{
                        stacked: true,
                        ticks: {
                            beginAtZero: true,
                        },
                        type: 'linear',
                    }]
                },
                plugins: {
                    zoom: {
                        zoom: {
                            drag: {
                                enabled: true,
                            }
                        }
                    },
                    legend: {
                        display: false
                     },
                     tooltips: {
                        enabled: false
                     },
                },
                responsive: true,
                maintainAspectRatio: false,
            }
        });
}