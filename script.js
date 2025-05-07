let model;
let lossData = [];

async function entrenarModelo() {
  const xs = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [8, 1]);
  const ys = tf.tensor2d([3, 5, 7, 9, 11, 13, 15, 17], [8, 1]);

  model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  lossData = [];
  await model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        lossData.push({ epoch, loss: logs.loss });
      }
    }
  });

  graficarPerdida();
}

function graficarPerdida() {
  const epochs = lossData.map(d => d.epoch);
  const losses = lossData.map(d => d.loss);

  const trace = {
    x: epochs,
    y: losses,
    type: 'scatter',
    mode: 'lines+markers',
    marker: { color: 'blue' },
    name: 'Loss'
  };

  const layout = {
    title: 'Evolución de la Pérdida (Loss)',
    xaxis: { title: 'Época' },
    yaxis: { title: 'Pérdida' },
    margin: { t: 30 }
  };

  Plotly.newPlot('lossPlot', [trace], layout);

  const inicial = losses[0].toFixed(4);
  const final = losses.at(-1).toFixed(4);
  const reduccion = ((1 - final / inicial) * 100).toFixed(2);
  document.getElementById("lossSummary").textContent =
    `Pérdida inicial: ${inicial}, final: ${final} (Reducción: ${reduccion}%)`;
}

function predecir() {
  const input = document.getElementById("inputX").value;
  const valores = input.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

  if (!model || valores.length === 0) {
    document.getElementById("resultado").innerHTML = "Modelo no entrenado o entrada inválida.";
    return;
  }

  const inputTensor = tf.tensor2d(valores, [valores.length, 1]);
  const outputTensor = model.predict(inputTensor);

  outputTensor.array().then(predicciones => {
    let html = "<strong>Estado:</strong> Modelo entrenado correctamente<br><br><strong>Resultados:</strong><ul>";
    valores.forEach((x, i) => {
      html += `<li>Para x = ${x}: y = ${predicciones[i][0].toFixed(2)}</li>`;
    });
    html += "</ul>";
    document.getElementById("resultado").innerHTML = html;
  });
}
