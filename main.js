const API = "http://127.0.0.1:8000/predict";

function showSingleResult(txt){
  document.getElementById("singleResult").innerText = txt;
}

function showBatchResult(txt){
  document.getElementById("batchResult").innerText = txt;
}

document.getElementById("predictBtn").addEventListener("click", async ()=>{
  const body = {
    wind_speed_10m: parseFloat(document.getElementById("wind_speed_10m").value||0),
    wind_speed_100m: parseFloat(document.getElementById("wind_speed_100m").value||0),
    wind_shear: parseFloat(document.getElementById("wind_shear").value||0),
    relative_humidity_2m: parseFloat(document.getElementById("relative_humidity_2m").value||0),
    cloud_cover: parseFloat(document.getElementById("cloud_cover").value||0),
    surface_pressure: parseFloat(document.getElementById("surface_pressure").value||0),
    dewpt_dep: parseFloat(document.getElementById("dewpt_dep").value||0)
  };
  showSingleResult("Waiting for API...");
  try{
    const res = await fetch(API, {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body)
    });
    const j = await res.json();
    if (res.ok){
      showSingleResult(`Prediction: ${j.predictions[0]}  —  Confidence: ${ (j.confidence[0]*100).toFixed(1) }%`);
    } else {
      showSingleResult("API error: " + (j.error || JSON.stringify(j)));
    }
  } catch(e){
    showSingleResult("Network / CORS error: " + e.message);
  }
});

document.getElementById("clearBtn").addEventListener("click", ()=>{
  ["wind_speed_10m","wind_speed_100m","wind_shear","relative_humidity_2m","cloud_cover","surface_pressure","dewpt_dep"].forEach(id=>{
    document.getElementById(id).value = "";
  });
  showSingleResult("");
});

function parseCSV(text){
  // tiny CSV parser (expects simple CSV, comma separated, header present)
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h=>h.trim());
  return lines.slice(1).map(l=>{
    const cols = l.split(',').map(c=>c.trim());
    const obj = {};
    headers.forEach((h,i)=> obj[h]= isNaN(cols[i]) ? cols[i] : parseFloat(cols[i]) );
    return obj;
  });
}

document.getElementById("uploadBtn").addEventListener("click", async ()=>{
  const f = document.getElementById("csvFile").files[0];
  if (!f){ showBatchResult("No file selected"); return; }
  showBatchResult("Reading file...");
  const text = await f.text();
  const rows = parseCSV(text);
  if (!rows.length){ showBatchResult("CSV parse failed or no rows"); return; }
  showBatchResult(`Parsed ${rows.length} rows — Sending to API...`);
  try{
    const res = await fetch(API, {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(rows)
    });
    const j = await res.json();
    if (res.ok){
      // j.predictions and j.confidence arrays
      const pairs = j.predictions.map((p,i)=> `${i+1}: ${p} (${(j.confidence[i]*100).toFixed(1)}%)` );
      showBatchResult(pairs.join("\n"));
    } else {
      showBatchResult("API error: " + (j.error || JSON.stringify(j)));
    }
  } catch(e){
    showBatchResult("Network / CORS error: " + e.message);
  }
});
