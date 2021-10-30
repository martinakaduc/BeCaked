function showGeoJSONData(map, filename) {
    // Create GeoJSON reader which will download the specified file.
    // Shape of the file was obtained by using HERE Geocoder API.
    // It is possible to customize look and feel of the objects.
    var reader = new H.data.geojson.Reader(filename, {
        // This function is called each time parser detects a new map object
        style: function (mapObject) {
            // Parsed geo objects could be styled using setStyle method
            if (mapObject instanceof H.map.Polygon) {
                mapObject.setStyle({
                    // fillColor: 'rgba(255, 0, 0, 0.5)',
                    fillColor: 'rgba(192, 57, 43,1.0)',
                    strokeColor: 'rgba(0, 0, 255, 0.2)',
                    lineWidth: 3
                });
            }
            mapObject.addEventListener('pointermove', () => { console.log(mapObject['data']['properties']['NAME_3'] + '\t' + mapObject['data']['properties']['NAME_2']) });
        }
    });

    // Start parsing the file
    reader.parse();

    // Add layer which shows GeoJSON data on the map
    map.addLayer(reader.getLayer());
}

var bubble;

function showInfo(ui, centre, label, data) {
    var arr_I = data['I']['real']
    var sum_I = arr_I.reduce((acc,ele)=>acc+ele,0)

    var arr_V = data['V']['real']
    var sum_V = arr_V.reduce((acc,ele)=>acc+ele,0)

    text = '<div style="font-weight: bold">'+label+'</div>'
    text += '<div> Confirmed cases: '+sum_I.toLocaleString()+'</div>'
    text += '<div> First dose: '+sum_V.toLocaleString()+'</div>'
    bubble = new H.ui.InfoBubble({ lat: centre[0], lng: centre[1] }, {
        content: text
    });

    // Add info bubble to the UI:
    ui.addBubble(bubble);
}

function hideInfo(){
    bubble.close()
}

var circles = []
function addCircleToMap(map, ui, centre, label, data, type, color1, color2) {
    var arr = data[type]['real']
    var sum = arr.reduce((acc,ele)=>acc+ele,0)
    var r = Math.sqrt(sum/ 3.14) * 50
    var circle = new H.map.Circle(
        // The central point of the circle
        { lat: centre[0], lng: centre[1] },
        // The radius of the circle in meters
        r,
        {
            style: {
                strokeColor: color1, // Color of the perimeter
                lineWidth: 2,
                fillColor: color2  // Color of the circle
            }
        }
    )
    map.addObject(circle);
    circles.push(circle);
    circle.addEventListener('pointerenter', () => { showInfo(ui, centre, label, data) });
    circle.addEventListener('pointerleave', () => { hideInfo() });
}

function removeAllCircle(map){
    circles.forEach(circle => {
        circle.setVisibility(false)
    });
}