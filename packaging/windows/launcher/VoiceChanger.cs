// Lanzador del Cambiador de voz portátil.
// Arranca el servidor (runtime\python.exe) en un puerto libre, espera a que responda, abre la
// interfaz en una ventana propia de Edge y apaga el servidor cuando esa ventana se cierra.
// Compilar: csc /target:winexe /out:VoiceChanger.exe /r:System.Windows.Forms.dll /r:System.Drawing.dll VoiceChanger.cs

using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Windows.Forms;

static class Launcher
{
    static string Base = AppDomain.CurrentDomain.BaseDirectory;
    static string DataDir = Path.Combine(Base, "datos");
    static string PortFile = Path.Combine(DataDir, "puerto.txt");
    static string LogFile = Path.Combine(DataDir, "registro_servidor.txt");
    static Process server;
    static StreamWriter log;

    [STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();
        Directory.CreateDirectory(DataDir);

        bool first;
        using (var mutex = new Mutex(true, "VoiceChangerPortable_" + Base.ToLowerInvariant().GetHashCode().ToString("X8"), out first))
        {
            if (!first)
            {
                // ya está en marcha: solo abrir otra ventana
                int p;
                if (File.Exists(PortFile) && int.TryParse(File.ReadAllText(PortFile).Trim(), out p)) OpenWindow(p, false);
                return;
            }
            try { Run(); }
            finally { StopServer(); try { File.Delete(PortFile); } catch { } }
        }
    }

    static void Run()
    {
        string python = Path.Combine(Base, @"runtime\python.exe");
        string serverDir = Path.Combine(Base, @"app\server");
        if (!File.Exists(python) || !Directory.Exists(serverDir))
        {
            MessageBox.Show("Faltan archivos del programa (runtime o app). Vuelve a descomprimir la carpeta completa.", "Cambiador de voz", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }
        if (Base.StartsWith(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), StringComparison.OrdinalIgnoreCase))
        {
            MessageBox.Show("No pongas esta carpeta dentro de «Archivos de programa»: el programa necesita guardar tus voces y ajustes en ella. Muévela, por ejemplo, a Documentos o al Escritorio.", "Cambiador de voz", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        // primera vez: el servidor descarga ~1.8 GB de modelos base a app\server\pretrain
        string pretrain = Path.Combine(serverDir, "pretrain");
        bool firstRun = !File.Exists(Path.Combine(pretrain, "hubert_base.pt")) || !File.Exists(Path.Combine(pretrain, "rmvpe.onnx"));
        if (firstRun && !Online())
        {
            MessageBox.Show("La primera vez hace falta internet para descargar los modelos base (unos 1,8 GB).\nConéctate a internet y vuelve a abrir el programa. Después ya funciona sin conexión.", "Cambiador de voz", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }

        string gpuWarning = CheckGpu();
        int port = FreePort();
        File.WriteAllText(PortFile, port.ToString());

        var splash = new Splash();
        splash.Show();
        splash.SetStatus("Arrancando el cambiador de voz…", gpuWarning);
        Application.DoEvents();

        log = new StreamWriter(LogFile, false, new UTF8Encoding(false)) { AutoFlush = true };
        var psi = new ProcessStartInfo(python, "-s -E -u -X utf8 MMVCServerSIO.py -p " + port + " --https false")
        {
            WorkingDirectory = serverDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        string torchLib = Path.Combine(Base, @"runtime\Lib\site-packages\torch\lib");
        psi.EnvironmentVariables["PATH"] = Path.Combine(Base, "runtime") + ";" + torchLib + ";" + Environment.GetEnvironmentVariable("PATH");
        psi.EnvironmentVariables["VC_DFN_MODEL_DIR"] = Path.Combine(Base, @"modelos\DeepFilterNet3");
        // el servidor no intenta abrir su propio cliente nativo: la ventana la abre este lanzador
        psi.EnvironmentVariables["VC_LAUNCHER"] = "1";
        server = new Process { StartInfo = psi, EnableRaisingEvents = true };
        bool downloadFailed = false;
        DataReceivedEventHandler onLine = (s, e) =>
        {
            if (e.Data == null) return;
            lock (log) log.WriteLine(e.Data);
            if (e.Data.IndexOf("failed to download weight", StringComparison.OrdinalIgnoreCase) >= 0) downloadFailed = true;
        };
        server.OutputDataReceived += onLine;
        server.ErrorDataReceived += onLine;
        server.Start();
        server.BeginOutputReadLine();
        server.BeginErrorReadLine();

        // esperar a que responda (la primera vez descarga ~1.6 GB de modelos base)
        var started = DateTime.Now;
        while (true)
        {
            Application.DoEvents();
            if (splash.Cancelled) return;
            if (server.HasExited)
            {
                splash.Close();
                MessageBox.Show("El servidor se ha cerrado al arrancar. Mira el registro:\n" + LogFile, "Cambiador de voz", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            if (downloadFailed)
            {
                splash.Close();
                StopServer();  // no dejar el servidor vivo detrás del aviso
                MessageBox.Show("No se pudieron descargar los modelos base. Hace falta internet la primera vez.\nComprueba la conexión y vuelve a abrir el programa.\n\nRegistro: " + LogFile, "Cambiador de voz", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            if (Responds(port)) break;
            string extra = gpuWarning + (gpuWarning.Length > 0 ? "\n" : "") + "Tiempo: " + (int)(DateTime.Now - started).TotalSeconds + " s";
            if (firstRun)
            {
                long got = DirSize(pretrain);
                const double total = 1769744629.0;
                int pct = (int)Math.Min(100, got * 100 / total);
                splash.SetStatus("Descargando los modelos base (solo la primera vez)…",
                                 string.Format("{0:0.0} de 1,8 GB ({1}%)", got / 1e9, pct) + "\n" + extra);
            }
            else splash.SetStatus("Arrancando el cambiador de voz…", extra);
            Thread.Sleep(500);
        }
        splash.Close();

        if (!OpenWindow(port, true))
        {
            // sin Edge: navegador normal y una ventanita para apagar
            var keep = new KeepAlive(port);
            Application.Run(keep);
        }
    }

    static bool OpenWindow(int port, bool waitForClose)
    {
        string url = "http://127.0.0.1:" + port + "/?mode=simple";
        string edge = FindEdge();
        if (edge == null)
        {
            Process.Start(url);
            return false;
        }
        var p = Process.Start(new ProcessStartInfo(edge,
            "--app=" + url + " --user-data-dir=\"" + Path.Combine(DataDir, "ventana") + "\" --no-first-run --no-default-browser-check --window-size=980,940")
        { UseShellExecute = false });
        if (waitForClose)
        {
            // con un perfil propio, este proceso vive mientras la ventana esté abierta
            while (!p.HasExited)
            {
                Application.DoEvents();
                if (server.HasExited) break;
                Thread.Sleep(300);
            }
        }
        return true;
    }

    static string FindEdge()
    {
        foreach (var root in new[] { Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles) })
        {
            string p = Path.Combine(root, @"Microsoft\Edge\Application\msedge.exe");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    static int FreePort()
    {
        // Windows (Hyper-V/WSL) reserva rangos enteros de puertos; probar hasta encontrar uno que se pueda abrir
        int[] candidates = { 18000, 18100, 18200, 18300, 18500, 17800, 17500, 16000 };
        foreach (int p in candidates) if (CanBind(p)) return p;
        var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        int port = ((IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        return port;
    }

    static bool CanBind(int port)
    {
        try { var l = new TcpListener(IPAddress.Loopback, port); l.Start(); l.Stop(); return true; }
        catch (SocketException) { return false; }
    }

    static long DirSize(string dir)
    {
        long n = 0;
        try { if (Directory.Exists(dir)) foreach (var f in Directory.GetFiles(dir, "*", SearchOption.AllDirectories)) { try { n += new FileInfo(f).Length; } catch { } } }
        catch { }
        return n;
    }

    static bool Online()
    {
        try
        {
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            var req = (HttpWebRequest)WebRequest.Create("https://huggingface.co/");
            req.Method = "HEAD";
            req.Timeout = 8000;
            using (var r = (HttpWebResponse)req.GetResponse()) return true;
        }
        catch (WebException e) { return e.Response != null; }  // any HTTP answer means we are online
        catch { return false; }
    }

    static bool Responds(int port)
    {
        try
        {
            var req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + port + "/");
            req.Timeout = 1500;
            using (var r = (HttpWebResponse)req.GetResponse()) return r.StatusCode == HttpStatusCode.OK;
        }
        catch { return false; }
    }

    static string CheckGpu()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=name,driver_version --format=csv,noheader")
            { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true };
            using (var p = Process.Start(psi))
            {
                string o = p.StandardOutput.ReadToEnd().Trim();
                p.WaitForExit(5000);
                if (o.Length == 0) return "No se encontró una tarjeta NVIDIA: funcionará, pero muy lento.";
                string[] parts = o.Split('\n')[0].Split(',');
                double ver;
                if (parts.Length > 1 && double.TryParse(parts[1].Trim().Split('.')[0], out ver) && ver < 570)
                    return "Tu driver NVIDIA es antiguo (" + parts[1].Trim() + "). Actualízalo a la versión 570 o superior para usar la GPU.";
                return "";
            }
        }
        catch { return "No se encontró una tarjeta NVIDIA: funcionará, pero muy lento."; }
    }

    static bool stopped;
    static void StopServer()
    {
        if (stopped) return;
        stopped = true;
        try
        {
            if (server != null && !server.HasExited)
            {
                // matar el árbol entero (el servidor arranca un proceso hijo)
                var k = Process.Start(new ProcessStartInfo("taskkill", "/PID " + server.Id + " /T /F") { UseShellExecute = false, CreateNoWindow = true });
                k.WaitForExit(10000);
            }
        }
        catch { }
        try { if (log != null) log.Dispose(); } catch { }
    }

    class Splash : Form
    {
        Label title = new Label(), detail = new Label();
        public bool Cancelled;
        public Splash()
        {
            Text = "Cambiador de voz";
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false; MinimizeBox = true;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(460, 170);
            Font = new Font("Segoe UI", 10f);
            title.SetBounds(20, 18, 420, 28); title.Font = new Font("Segoe UI", 12f, FontStyle.Bold);
            detail.SetBounds(20, 52, 420, 60);
            var bar = new ProgressBar { Style = ProgressBarStyle.Marquee, MarqueeAnimationSpeed = 30 };
            bar.SetBounds(20, 118, 300, 22);
            var cancel = new Button { Text = "Cancelar" };
            cancel.SetBounds(340, 114, 100, 30);
            cancel.Click += (s, e) => { Cancelled = true; };
            Controls.AddRange(new Control[] { title, detail, bar, cancel });
            FormClosing += (s, e) => { Cancelled = true; };
        }
        public void SetStatus(string t, string d) { title.Text = t; detail.Text = d; }
    }

    class KeepAlive : Form
    {
        public KeepAlive(int port)
        {
            Text = "Cambiador de voz";
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(460, 130);
            Font = new Font("Segoe UI", 10f);
            var l = new Label { Text = "El cambiador está funcionando en tu navegador:\nhttp://127.0.0.1:" + port + "/\nCierra esta ventana para apagarlo." };
            l.SetBounds(20, 15, 420, 70);
            var b = new Button { Text = "Apagar" };
            b.SetBounds(340, 88, 100, 30);
            b.Click += (s, e) => Close();
            Controls.AddRange(new Control[] { l, b });
        }
    }
}
