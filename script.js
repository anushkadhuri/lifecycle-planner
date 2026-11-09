/* ---------- U3: past projects knowledge base ---------- */
const PAST = [
    {t:"Credit card fraud detection", d:"classify fraudulent card transactions from highly imbalanced tabular data using gradient boosting and precision recall", tip:"Use PR-AUC and resampling, accuracy is misleading on imbalanced classes."},
    {t:"Customer churn prediction for a bank", d:"predict customers who will leave the bank from account usage billing and demographic records", tip:"Time-based split avoids leakage, and explain top features to business teams."},
    {t:"House price regression", d:"predict house prices from tabular features such as area location and rooms using regression", tip:"Log-transform the target and check residuals per neighbourhood."},
    {t:"Pneumonia detection from chest X-rays", d:"classify chest x-ray images as normal or pneumonia using a convolutional neural network and transfer learning", tip:"Augment images, use recall as a key metric and review Grad-CAM heatmaps."},
    {t:"Plant disease image classifier", d:"identify crop leaf diseases from photos using a CNN with data augmentation", tip:"Photograph variety matters more than model size, so test on field images."},
    {t:"Sentiment analysis of product reviews", d:"classify review text as positive or negative using embeddings and a transformer", tip:"Hold out a different product category to test generalisation."},
    {t:"Hospital readmission risk", d:"predict patient readmission within 30 days from clinical records tabular health data", tip:"Remove identifiers, check fairness across age and gender groups."},
    {t:"Handwritten digit recognition", d:"recognise handwritten digits from images with a small convolutional network on MNIST", tip:"A small CNN reaches high accuracy fast, so keep the plan lightweight."},
    {t:"Loan default scoring", d:"predict whether a borrower will default using income credit history and loan features", tip:"Audit for bias on protected attributes and keep the model explainable."},
    {t:"Sales demand forecasting", d:"forecast weekly product demand from historical sales time series data", tip:"Use rolling-window validation, never random splits."},
    {t:"Brain tumour MRI segmentation", d:"segment tumour regions in MRI scans using a U-Net deep learning model", tip:"Dice score is the main metric, and expert-annotated labels are costly."},
    {t:"Spam email filter", d:"classify emails as spam or not spam using text features and naive bayes", tip:"Retrain regularly, since spam patterns drift."}
  ];
  const STOP = new Set("a an the of to and or in on for from with using use is are be will by as at it this that which who we our their can into than then them its".split(" "));
  const tok = s => s.toLowerCase().replace(/[^a-z0-9\s-]/g," ").split(/\s+/).filter(w=>w.length>2 && !STOP.has(w));
  
  function retrieve(query, k=3){
    const docs = PAST.map(p=>tok(p.t+" "+p.d));
    const q = tok(query);
    const all = docs.concat([q]);
    const df = {};
    all.forEach(d=>new Set(d).forEach(w=>df[w]=(df[w]||0)+1));
    const N = all.length;
    const vec = d=>{const v={};d.forEach(w=>v[w]=(v[w]||0)+1);Object.keys(v).forEach(w=>v[w]*=Math.log((N+1)/(df[w]+0.5))); return v;};
    const qv = vec(q);
    const norm = v=>Math.sqrt(Object.values(v).reduce((a,b)=>a+b*b,0))||1;
    return PAST.map((p,i)=>{
      const dv = vec(docs[i]);
      let dot=0; for(const w in qv) if(dv[w]) dot+=qv[w]*dv[w];
      return {...p, score: dot/(norm(qv)*norm(dv))};
    }).sort((a,b)=>b.score-a.score).slice(0,k);
  }
  
  /* ---------- U1: role + output-format prompts per stage ---------- */
  const STAGES = [
    {id:"S1", name:"Problem definition", role:"You are a senior data science project lead.", format:"Return 4 bullets: objective, target variable, success metric, constraints."},
    {id:"S2", name:"Data collection", role:"You are a data engineer.", format:"Return 4 bullets: sources, volume needed, labelling method, storage."},
    {id:"S3", name:"Exploration and cleaning", role:"You are a data analyst.", format:"Return 4 bullets: checks, missing values, outliers, visualisations."},
    {id:"S4", name:"Features and preprocessing", role:"You are a feature engineering specialist.", format:"Return 4 bullets: transformations, encoding, scaling, split strategy."},
    {id:"S5", name:"Model building", role:"You are an ML engineer.", format:"Return 4 bullets: baseline, main model, training setup, tuning."},
    {id:"S6", name:"Evaluation", role:"You are a model validation reviewer.", format:"Return 4 bullets: metrics, validation method, error analysis, acceptance threshold."},
    {id:"S7", name:"Deployment", role:"You are an MLOps engineer.", format:"Return 4 bullets: serving method, interface, versioning, rollback."},
    {id:"S8", name:"Monitoring", role:"You are a reliability engineer.", format:"Return 4 bullets: drift checks, performance tracking, alerts, retraining trigger."}
  ];
  
  /* ---------- U5: privacy and bias checkpoints ---------- */
  const CHK = {
    S1:{p:"Confirm the legal basis and consent for using personal data in this project.", b:"Check that the goal does not disadvantage any group of people."},
    S2:{p:"Collect only the fields you need and record who owns the data.", b:"Check that the sample represents all user groups, not only the easiest to reach."},
    S3:{p:"Mask or remove names, IDs and contact details before sharing analysis.", b:"Compare missing data and label quality across groups."},
    S4:{p:"Drop or hash direct identifiers, and avoid proxies for sensitive attributes.", b:"Split data so no group is missing from validation or test sets."},
    S5:{p:"Train on anonymised data, and keep raw data out of logs and notebooks.", b:"Try class weights or reweighting if groups are unbalanced."},
    S6:{p:"Report results without exposing individual records.", b:"Report metrics per group, and flag gaps larger than 5 percentage points."},
    S7:{p:"Encrypt data in transit, set access control, and delete user inputs that are not needed.", b:"Give users a way to question or appeal a model decision."},
    S8:{p:"Set data retention limits for monitoring logs.", b:"Re-check group-wise performance after every retraining."}
  };
  
  /* ---------- Stage drafting (rule-based stand-in for the GPT draft) ---------- */
  function draft(stage, problem, type, refs){
    const short = problem.trim().replace(/\.$/,"");
    const dl = type==="DL";
    const unit = dl ? "images" : "records";
    const model = dl ? "a CNN with transfer learning (e.g. MobileNetV2)" : "a dense neural network, compared with a gradient boosting baseline";
    const metric = dl ? "recall, F1 and AUC (plus accuracy as a reference)" : "F1, ROC-AUC and precision-recall";
    const lesson = refs[0] ? refs[0].tip : "Document every assumption.";
    const m = {
      S1:[`Objective: ${short}.`,`Target variable: define the label clearly and how it is measured.`,`Success metric: ${metric}, agreed with the stakeholder.`,`Constraints: data access, compute budget, and the deadline.`],
      S2:[`Sources: internal databases, public datasets, or manual collection of ${unit}.`,`Volume: start with at least ${dl?"2,000 labelled images per class":"5,000 labelled records"}.`,`Labelling: ${dl?"expert annotation with a second reviewer":"derive labels from historical outcomes"}.`,`Storage: versioned folder or table with a data dictionary.`],
      S3:[`Check shape, types, duplicates and class balance.`,`Missing values: ${dl?"remove unreadable or blank images":"impute or flag, depending on the column"}.`,`Outliers: ${dl?"inspect extreme sizes, brightness and mislabelled samples":"use IQR or z-score checks"}.`,`Visualise: ${dl?"sample grids per class and pixel statistics":"histograms, correlation heatmap, target vs feature plots"}.`],
      S4:[dl?`Resize to 224x224 and normalise pixel values to 0-1.`:`Encode categories and scale numeric columns.`,dl?`Augment with flips, small rotations and brightness changes.`:`Create ratio and aggregate features from raw columns.`,`Split train/validation/test as 70/15/15, stratified by label.`,`Lesson from similar projects: ${lesson}`],
      S5:[`Baseline: ${dl?"a small CNN trained from scratch":"logistic regression"}.`,`Main model: ${model}.`,`Training: Adam optimiser, early stopping, batch size ${dl?32:64}.`,`Tuning: learning rate, layer size and dropout via a small search.`],
      S6:[`Metrics: ${metric}.`,`Validation: ${dl?"hold-out test set plus a few real-world samples":"stratified k-fold (k=5) plus a final test set"}.`,`Error analysis: review false positives and false negatives${dl?" with Grad-CAM":""}.`,`Acceptance threshold: beat the baseline by a margin agreed up front.`],
      S7:[`Serve with ${dl?"a Streamlit or Flask app that accepts an image":"a Streamlit or Flask form or REST endpoint"}.`,`Package the model with a version tag and a requirements file.`,`Interface: show the prediction with a confidence score.`,`Rollback: keep the previous model version ready to switch back.`],
      S8:[`Track input drift against training data weekly.`,`Log predictions and, when labels arrive, live accuracy.`,`Alert when a metric falls more than 5% below the baseline.`,`Retrain when drift persists or after a scheduled period.`]
    };
    return m[stage.id];
  }
  
  /* ---------- U2: prompt chaining ---------- */
  function runChain(problem, type, refs){
    let carry = "";
    return STAGES.map(st=>{
      const prompt = `${st.role}\nProblem: ${problem}\nProject type: ${type}\nPrevious stage output:\n${carry||"(none)"}\n\nTask: plan the "${st.name}" stage.\n${st.format}\nAlso list one privacy and one bias checkpoint.`;
      const items = draft(st, problem, type, refs);
      carry = items.join(" ");
      return {...st, prompt, items, chk:CHK[st.id]};
    });
  }
  
  /* Extractive summary: first bullet of each stage */
  function summarise(plan){
    return plan.map(s=>`${s.name}: ${s.items[0].replace(/\.$/,"")}.`).join(" ");
  }
  
  /* ---------- Mermaid diagram ---------- */
  function buildMermaid(plan){
    let g = "flowchart TD\n  START([Problem statement]) --> S1\n";
    plan.forEach((s,i)=>{
      g += `  ${s.id}["${i+1}. ${s.name}"]\n`;
      g += `  C${s.id}{{"Privacy + bias check"}}\n`;
      g += `  ${s.id} --> C${s.id}\n`;
      const next = plan[i+1] ? plan[i+1].id : "END";
      g += `  C${s.id} --> ${next}\n`;
    });
    g += "  END([Deployed and monitored model])\n  S8 -.->|drift detected| S2\n";
    g += "  classDef chk stroke-dasharray: 5 4\n  class " + plan.map(s=>"C"+s.id).join(",") + " chk\n";
    return g;
  }
  
  /* ---------- U4: notebook generator (TensorFlow 2) ---------- */
  function buildNotebook(problem, type){
    const dl = type==="DL";
    const md = t=>({cell_type:"markdown",metadata:{},source:t.split("\n").map((l,i,a)=>l+(i<a.length-1?"\n":""))});
    const code = t=>({cell_type:"code",metadata:{},execution_count:null,outputs:[],source:t.split("\n").map((l,i,a)=>l+(i<a.length-1?"\n":""))});
    const cells = [
      md(`# Starter notebook: ${type} project\n**Problem:** ${problem}\n\nGenerated by the ML/DL Lifecycle Planner. Replace the TODO parts with your data.`),
      code(`import numpy as np\nimport pandas as pd\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow.keras import layers\nprint("TensorFlow", tf.__version__)`),
      md(`## 1. Load data`)
    ];
    if(dl){
      cells.push(code(`# TODO: put images in data/train/<class_name>/*.jpg and data/val/<class_name>/*.jpg\nIMG = (224, 224)\ntrain_ds = keras.utils.image_dataset_from_directory("data/train", image_size=IMG, batch_size=32)\nval_ds = keras.utils.image_dataset_from_directory("data/val", image_size=IMG, batch_size=32)\nclass_names = train_ds.class_names\nprint(class_names)`));
      cells.push(md(`## 2. Augmentation and model (transfer learning)`));
      cells.push(code(`aug = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.05), layers.RandomZoom(0.1)])\nbase = keras.applications.MobileNetV2(input_shape=IMG+(3,), include_top=False, weights="imagenet")\nbase.trainable = False\n\ninputs = keras.Input(shape=IMG+(3,))\nx = aug(inputs)\nx = keras.applications.mobilenet_v2.preprocess_input(x)\nx = base(x, training=False)\nx = layers.GlobalAveragePooling2D()(x)\nx = layers.Dropout(0.3)(x)\noutputs = layers.Dense(len(class_names), activation="softmax")(x)\nmodel = keras.Model(inputs, outputs)\nmodel.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])\nmodel.summary()`));
      cells.push(md(`## 3. Train`));
      cells.push(code(`early = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)\nhistory = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early])`));
    } else {
      cells.push(code(`# TODO: replace with your file. The last column must be the target.\ndf = pd.read_csv("data.csv")\nprint(df.shape)\ndf.head()`));
      cells.push(md(`## 2. Preprocess`));
      cells.push(code(`from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n\ntarget = df.columns[-1]\nX = pd.get_dummies(df.drop(columns=[target])).fillna(0).astype("float32")\ny = df[target].astype("int32")\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)`));
      cells.push(md(`## 3. Model and training`));
      cells.push(code(`model = keras.Sequential([\n    layers.Input(shape=(X_train.shape[1],)),\n    layers.Dense(64, activation="relu"),\n    layers.Dropout(0.3),\n    layers.Dense(32, activation="relu"),\n    layers.Dense(1, activation="sigmoid"),\n])\nmodel.compile(optimizer="adam", loss="binary_crossentropy",\n              metrics=["accuracy", keras.metrics.AUC(name="auc")])\nearly = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)\nhistory = model.fit(X_train, y_train, validation_split=0.15, epochs=50, batch_size=64, callbacks=[early])`));
    }
    cells.push(md(`## 4. Evaluate`));
    cells.push(code(dl
      ? `val_loss, val_acc = model.evaluate(val_ds)\nprint("Validation accuracy:", round(val_acc, 3))\n# TODO: add per-class recall and a confusion matrix`
      : `from sklearn.metrics import classification_report\nprobs = model.predict(X_test).ravel()\nprint(classification_report(y_test, (probs > 0.5).astype(int)))`));
    cells.push(md(`## 5. Privacy and bias checkpoints\n- [ ] Identifiers removed or hashed before training\n- [ ] Metrics reported per group\n- [ ] Gaps larger than 5 points reviewed`));
    cells.push(code(`model.save("model.keras")`));
    return {cells, metadata:{kernelspec:{display_name:"Python 3",language:"python",name:"python3"}}, nbformat:4, nbformat_minor:5};
  }
  
  /* ---------- Cover image ---------- */
  function drawCover(problem, type){
    const c = document.getElementById("cover"), x = c.getContext("2d");
    const W = c.width, H = c.height;
    const dark = type==="DL";
    const bg = x.createLinearGradient(0,0,W,H);
    bg.addColorStop(0, dark?"#10303A":"#12343B"); bg.addColorStop(1, dark?"#2B1F40":"#0E7C86");
    x.fillStyle = bg; x.fillRect(0,0,W,H);
    const layers = dark ? [3,5,5,4,2] : [4,6,4,1];
    const pts = layers.map((n,i)=>Array.from({length:n},(_,j)=>({x:520+i*(360/(layers.length-1)), y:60+(j+0.5)*(300/n)})));
    x.strokeStyle = "rgba(255,255,255,0.18)"; x.lineWidth = 1;
    for(let i=0;i<pts.length-1;i++) pts[i].forEach(a=>pts[i+1].forEach(b=>{x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke();}));
    pts.flat().forEach(p=>{x.fillStyle="#EAF7F8";x.beginPath();x.arc(p.x,p.y,6,0,7);x.fill();});
    x.fillStyle = "#FFFFFF"; x.font = "600 40px 'IBM Plex Sans', sans-serif"; x.textBaseline = "top";
    x.fillText(type==="DL"?"Deep learning plan":"Machine learning plan", 40, 50);
    x.font = "400 22px 'IBM Plex Sans', sans-serif"; x.fillStyle = "rgba(255,255,255,0.88)";
    wrap(x, problem, 40, 120, 440, 30, 6);
    x.font = "400 16px 'IBM Plex Mono', monospace"; x.fillStyle = "rgba(255,255,255,0.7)";
    x.fillText("8 stages | privacy + bias checks at every stage", 40, H-50);
  }
  function wrap(x, text, px, py, maxW, lh, maxLines){
    const words = text.split(/\s+/); let line = "", n = 0;
    for(const w of words){
      const t = line ? line+" "+w : w;
      if(x.measureText(t).width > maxW){ x.fillText(line, px, py+n*lh); line = w; n++; if(n>=maxLines) return; }
      else line = t;
    }
    if(line) x.fillText(line, px, py+n*lh);
  }
  
  /* ---------- UI ---------- */
  mermaid.initialize({startOnLoad:false, securityLevel:"loose",
    theme: matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "default"});
  const $ = id => document.getElementById(id);
  let state = {};
  
  const PRESETS = {
    ml:{type:"ML", text:"Predict which telecom customers will cancel their plan next month using billing, usage and support-call records."},
    dl:{type:"DL", text:"Classify chest X-ray images as normal or pneumonia to help radiologists prioritise urgent scans."}
  };
  document.querySelectorAll("[data-preset]").forEach(b=>b.onclick=()=>{
    const p = PRESETS[b.dataset.preset];
    $("problem").value = p.text; $("ptype").value = p.type; generate();
  });
  $("go").onclick = generate;
  
  function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
  
  async function generate(){
    const problem = $("problem").value.trim();
    if(!problem){ $("problem").focus(); return; }
    const type = $("ptype").value;
    const refs = retrieve(problem + " " + (type==="DL" ? "image convolutional neural network" : "tabular records"));
    const plan = runChain(problem, type, refs);
    const mmd = buildMermaid(plan);
    state = {problem, type, plan, mmd};
  
    $("summary").textContent = summarise(plan);
    $("refs").innerHTML = refs.map(r=>`<div class="ref"><div><strong>${esc(r.t)}</strong><br><span style="color:var(--muted);font-size:.9rem">${esc(r.tip)}</span></div><div class="score">${r.score.toFixed(2)}</div></div>`).join("");
    $("stages").innerHTML = plan.map((s,i)=>`
      <div class="stage">
        <h3>${i+1}. ${esc(s.name)}</h3>
        <div class="role">${esc(s.role)}</div>
        <ul>${s.items.map(t=>`<li>${esc(t)}</li>`).join("")}</ul>
        <div class="chk p"><strong>Privacy:</strong> ${esc(s.chk.p)}</div>
        <div class="chk b"><strong>Bias:</strong> ${esc(s.chk.b)}</div>
        <details><summary>Prompt used for this stage</summary><pre>${esc(s.prompt)}</pre></details>
      </div>`).join("");
  
    $("out").style.display = "grid";
    try{
      const {svg} = await mermaid.render("g"+Date.now(), mmd);
      $("diagram").innerHTML = svg; state.svg = svg;
    }catch(e){ $("diagram").innerHTML = "<pre>"+esc(mmd)+"</pre>"; }
    drawCover(problem, type);
    $("out").scrollIntoView({behavior:"smooth"});
  }
  
  function download(name, data, mime){
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([data],{type:mime}));
    a.download = name; a.click(); URL.revokeObjectURL(a.href);
  }
  $("dlmmd").onclick = ()=>download("lifecycle_"+state.type+".mmd", state.mmd, "text/plain");
  $("dlsvg").onclick = ()=>state.svg && download("lifecycle_"+state.type+".svg", state.svg, "image/svg+xml");
  $("dlnb").onclick = ()=>download("starter_"+state.type+".ipynb", JSON.stringify(buildNotebook(state.problem, state.type), null, 1), "application/json");
  $("dlpng").onclick = ()=>{const a=document.createElement("a");a.href=$("cover").toDataURL("image/png");a.download="cover_"+state.type+".png";a.click();};