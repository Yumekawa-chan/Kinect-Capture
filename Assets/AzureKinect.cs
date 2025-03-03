using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.Azure.Kinect.Sensor;
using System.Threading.Tasks;

public class MultiKinectScript : MonoBehaviour
{
    // 両デバイスで共通の点群マテリアル（Inspector から設定可能）
    public Material pointCloudMaterial;

    // 回転行列の扱い（true: 転置版を使用、false: JSONにあるそのままの行列）
    public bool useTransposedRotation = true;

    // 点のサイズ設定（Inspector から調整可能）
    public float pointSize = 5.0f;

    // 手動調整用パラメータ（Inspector から調整可能）
    public Vector3 kinect2ManualPosition = new Vector3(-0.39f, 0.11f, -1.13f);
    public Vector3 kinect2ManualRotation = new Vector3(0.167f, -0.801f, -4.548f);

    // Kinect デバイス
    private Device kinect1;
    private Device kinect2;

    // Kinect1 用点群 Mesh とデータ
    private Mesh mesh1;
    private int num1;
    private Vector3[] vertices1;
    private Color32[] colors1;
    private int[] indices1;

    // Kinect2 用点群 Mesh とデータ
    private Mesh mesh2;
    private int num2;
    private Vector3[] vertices2;
    private Color32[] colors2;
    private int[] indices2;

    // 各 Kinect の Transformation（SDK による内部キャリブレーション情報）
    private Transformation transformation1;
    private Transformation transformation2;

    // Kinect2→Kinect1 への外部校正パラメータ（JSON）の Unity 用変換行列
    // ※ここでは、まず Kinect2 の raw 点群（ミリ→メートル変換のみ済み）に対して T_ext を適用し、
    //    その後 Kinect1 と同じように y 軸反転する、という流れとします。
    private Matrix4x4 transform2To1;

    // Kinect2 用点群表示オブジェクト
    private GameObject kinect2Object;

    void Start()
    {
        // マテリアルが未設定ならデフォルトを作成
        if (pointCloudMaterial == null)
        {
            // ポイントクラウド表示用のカスタムシェーダーを使用したマテリアルを作成
            Shader shader = Shader.Find("Custom/ColoredVertex");
            if (shader != null)
            {
                pointCloudMaterial = new Material(shader);
                pointCloudMaterial.SetFloat("_PointSize", pointSize); // 点のサイズを設定
                pointCloudMaterial.SetFloat("_Size", 1.0f);           // スケーリング係数
            }
            else
            {
                Debug.LogError("Custom/ColoredVertex シェーダーが見つかりませんでした。Assets/ColoredVertex.shaderを確認してください。");

                // フォールバック: 標準シェーダーを試す
                shader = Shader.Find("Standard");
                if (shader != null)
                {
                    Debug.LogWarning("フォールバックとしてStandardシェーダーを使用します（点の表示が最適化されません）");
                    pointCloudMaterial = new Material(shader);
                }
                else
                {
                    Debug.LogError("シェーダーが見つかりませんでした。マテリアルを手動で設定してください。");
                    return;
                }
            }
        }
        else
        {
            // 既存マテリアルに対してもPointSizeを設定
            pointCloudMaterial.SetFloat("_PointSize", pointSize);
        }

        // デバイス初期化
        InitKinectDevices();

        // Mesh 初期化
        InitMesh1();
        InitMesh2();

        // 外部校正パラメータを元に変換行列をセットアップ
        SetupExternalTransformation();

        // 非同期で各デバイスの点群更新ループを開始
        Task t1 = KinectLoop1();
        Task t2 = KinectLoop2();
    }

    // Kinect の初期化（両デバイス）
    private void InitKinectDevices()
    {
        // Kinect1（デバイス0）の初期化
        kinect1 = Device.Open(0);
        kinect1.StartCameras(new DeviceConfiguration
        {
            ColorFormat = ImageFormat.ColorBGRA32,
            ColorResolution = ColorResolution.R720p,
            DepthMode = DepthMode.NFOV_2x2Binned,
            SynchronizedImagesOnly = true,
            CameraFPS = FPS.FPS30
        });
        transformation1 = kinect1.GetCalibration().CreateTransformation();

        // Kinect2（デバイス1）の初期化
        kinect2 = Device.Open(1);
        kinect2.StartCameras(new DeviceConfiguration
        {
            ColorFormat = ImageFormat.ColorBGRA32,
            ColorResolution = ColorResolution.R720p,
            DepthMode = DepthMode.NFOV_2x2Binned,
            SynchronizedImagesOnly = true,
            CameraFPS = FPS.FPS30
        });
        transformation2 = kinect2.GetCalibration().CreateTransformation();
    }

    // Kinect1 用 Mesh の初期化（このスクリプトがアタッチされた GameObject に設定）
    private void InitMesh1()
    {
        int width = kinect1.GetCalibration().DepthCameraCalibration.ResolutionWidth;
        int height = kinect1.GetCalibration().DepthCameraCalibration.ResolutionHeight;
        num1 = width * height;

        mesh1 = new Mesh();
        mesh1.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        vertices1 = new Vector3[num1];
        colors1 = new Color32[num1];
        indices1 = new int[num1];
        for (int i = 0; i < num1; i++)
        {
            indices1[i] = i;
        }
        mesh1.vertices = vertices1;
        mesh1.colors32 = colors1;
        mesh1.SetIndices(indices1, MeshTopology.Points, 0);

        // MeshFilterとMeshRendererがなければ追加
        MeshFilter mf = gameObject.GetComponent<MeshFilter>();
        if (mf == null)
        {
            mf = gameObject.AddComponent<MeshFilter>();
        }
        mf.mesh = mesh1;

        MeshRenderer mr = gameObject.GetComponent<MeshRenderer>();
        if (mr == null)
        {
            mr = gameObject.AddComponent<MeshRenderer>();
        }
        mr.material = pointCloudMaterial;
    }

    // Kinect2 用 Mesh の初期化（新規 GameObject を生成）
    private void InitMesh2()
    {
        int width = kinect2.GetCalibration().DepthCameraCalibration.ResolutionWidth;
        int height = kinect2.GetCalibration().DepthCameraCalibration.ResolutionHeight;
        num2 = width * height;

        mesh2 = new Mesh();
        mesh2.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        vertices2 = new Vector3[num2];
        colors2 = new Color32[num2];
        indices2 = new int[num2];
        for (int i = 0; i < num2; i++)
        {
            indices2[i] = i;
        }
        mesh2.vertices = vertices2;
        mesh2.colors32 = colors2;
        mesh2.SetIndices(indices2, MeshTopology.Points, 0);

        kinect2Object = new GameObject("Kinect2_PointCloud");
        MeshFilter mf2 = kinect2Object.AddComponent<MeshFilter>();
        mf2.mesh = mesh2;
        MeshRenderer mr2 = kinect2Object.AddComponent<MeshRenderer>();
        mr2.material = pointCloudMaterial;

        // 手動調整した位置・回転を適用
        kinect2Object.transform.position = kinect2ManualPosition;
        kinect2Object.transform.eulerAngles = kinect2ManualRotation;

        Debug.Log($"Kinect2_PointCloudに手動調整を適用しました: 位置({kinect2ManualPosition.x}, {kinect2ManualPosition.y}, {kinect2ManualPosition.z}), 回転({kinect2ManualRotation.x}, {kinect2ManualRotation.y}, {kinect2ManualRotation.z})");
    }

    // 外部校正パラメータの Unity 用変換行列のセットアップ
    private void SetupExternalTransformation()
    {
        // JSON から得た外部パラメータ（OpenCV or Open3D 座標系前提とする）
        float r11 = 0.8107891645868455f;
        float r12 = 0.0019960699214603833f;
        float r13 = -0.5853349009698925f;
        float r21 = -0.06550264646595835f;
        float r22 = 0.9940224675077751f;
        float r23 = -0.08734264362675243f;
        float r31 = 0.5816617005567075f;
        float r32 = 0.10915745414135877f;
        float r33 = 0.8060733938735705f;
        // 平行移動（ミリ→メートル換算済み）
        float t_x = -1763.7806835019967f * 0.001f;
        float t_y = -354.33494633983526f * 0.001f;
        float t_z = 1164.4870327703366f * 0.001f;

        // 回転行列の構成（useTransposedRotation が true の場合は転置行列を使用）
        Matrix4x4 R = Matrix4x4.identity;
        if (useTransposedRotation)
        {
            // 転置版：各列を行として設定
            R.SetRow(0, new Vector4(r11, r21, r31, 0));
            R.SetRow(1, new Vector4(r12, r22, r32, 0));
            R.SetRow(2, new Vector4(r13, r23, r33, 0));
        }
        else
        {
            R.SetRow(0, new Vector4(r11, r12, r13, 0));
            R.SetRow(1, new Vector4(r21, r22, r23, 0));
            R.SetRow(2, new Vector4(r31, r32, r33, 0));
        }

        // 外部パラメータから作る変換行列 T_ext： p₁ = R * p₂ + t
        Matrix4x4 T_ext = Matrix4x4.identity;
        T_ext.SetRow(0, new Vector4(R[0, 0], R[0, 1], R[0, 2], t_x));
        T_ext.SetRow(1, new Vector4(R[1, 0], R[1, 1], R[1, 2], t_y));
        T_ext.SetRow(2, new Vector4(R[2, 0], R[2, 1], R[2, 2], t_z));

        // Kinect1 の点群は (x, -y, z) になっているので、
        // Kinect2 の raw 点群は (x, y, z)（ミリ→メートル変換のみ済み）の状態とし、
        // T_ext を適用した後、同様に y 軸反転して合わせる
        Matrix4x4 invertY = Matrix4x4.identity;
        invertY[1, 1] = -1;

        transform2To1 = invertY * T_ext;
    }

    // Kinect1 の点群取得ループ
    private async Task KinectLoop1()
    {
        while (true)
        {
            using (Capture capture = await Task.Run(() => kinect1.GetCapture()).ConfigureAwait(true))
            {
                // Kinect1 用：カラー画像を Depth カメラに合わせる
                Image colorImage = transformation1.ColorImageToDepthCamera(capture);
                BGRA[] colorArray = colorImage.GetPixels<BGRA>().ToArray();

                // Depth 画像から点群を取得し、(x, y, z) を (x, -y, z) に変換
                Image xyzImage = transformation1.DepthImageToPointCloud(capture.Depth);
                Short3[] xyzArray = xyzImage.GetPixels<Short3>().ToArray();

                for (int i = 0; i < num1; i++)
                {
                    vertices1[i] = new Vector3(
                        xyzArray[i].X * 0.001f,
                        -xyzArray[i].Y * 0.001f,
                        xyzArray[i].Z * 0.001f
                    );
                    colors1[i] = new Color32(
                        colorArray[i].R,
                        colorArray[i].G,
                        colorArray[i].B,
                        255
                    );
                }
                mesh1.vertices = vertices1;
                mesh1.colors32 = colors1;
                mesh1.RecalculateBounds();
            }
        }
    }

    // Kinect2 の点群取得ループ（外部変換を適用）
    private async Task KinectLoop2()
    {
        while (true)
        {
            using (Capture capture = await Task.Run(() => kinect2.GetCapture()).ConfigureAwait(true))
            {
                // Kinect2 用：カラー画像を Depth カメラ座標に合わせる
                Image colorImage = transformation2.ColorImageToDepthCamera(capture);
                BGRA[] colorArray = colorImage.GetPixels<BGRA>().ToArray();

                // Depth 画像から点群を取得（raw 値：ミリ→メートル変換のみ）
                Image xyzImage = transformation2.DepthImageToPointCloud(capture.Depth);
                Short3[] xyzArray = xyzImage.GetPixels<Short3>().ToArray();

                for (int i = 0; i < num2; i++)
                {
                    // Kinect2 の raw 点（ミリ→メートル換算のみ）
                    Vector3 rawPoint = new Vector3(
                        xyzArray[i].X * 0.001f,
                        xyzArray[i].Y * 0.001f,
                        xyzArray[i].Z * 0.001f
                    );
                    // 外部校正変換を適用後、Kinect1 と同様に y 軸反転して合わせる
                    vertices2[i] = transform2To1.MultiplyPoint3x4(rawPoint);

                    colors2[i] = new Color32(
                        colorArray[i].R,
                        colorArray[i].G,
                        colorArray[i].B,
                        255
                    );
                }
                mesh2.vertices = vertices2;
                mesh2.colors32 = colors2;
                mesh2.RecalculateBounds();
            }
        }
    }

    // アプリ終了時に各 Kinect のカメラを停止
    private void OnDestroy()
    {
        if (kinect1 != null)
        {
            kinect1.StopCameras();
            kinect1.Dispose();
        }
        if (kinect2 != null)
        {
            kinect2.StopCameras();
            kinect2.Dispose();
        }
    }
}

