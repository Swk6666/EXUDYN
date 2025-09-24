#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

struct FlexibleLinkParameters {
    double length_first = 1.0;
    double length_second = 0.8;
    double width = 0.012;
    double thickness = 0.003;
    double youngs_modulus = 5.0e9;
    double density = 1180.0;
    double poisson = 0.34;
    int modes_first = 16;
    int modes_second = 16;
    double mesh_maxh = 0.01;
    double mass_proportional_damping = 2e-3;
    double stiffness_proportional_damping = 8e-3;
    double gravity = 9.81;
};

struct InitialState {
    double theta1 = M_PI / 4.0;
    double theta2 = -35.0 * M_PI / 180.0;
    double omega1 = 0.6;
    double omega2 = -0.8;
};

struct SimulationSetup {
    double end_time = 20.0;
    double step_size = 1e-3;
    bool store_trajectory = true;
    int frames_per_second = 60;
};

static py::array_t<double> rotation_z(double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    py::array_t<double> mat({3, 3});
    auto buf = mat.mutable_unchecked<2>();
    buf(0, 0) = c;  buf(0, 1) = -s; buf(0, 2) = 0.0;
    buf(1, 0) = s;  buf(1, 1) =  c; buf(1, 2) = 0.0;
    buf(2, 0) = 0.0;buf(2, 1) = 0.0;buf(2, 2) = 1.0;
    return mat;
}

static std::array<double, 3> to_vector3(const py::array& arr) {
    if (arr.ndim() != 1 || arr.shape(0) != 3) {
        throw std::runtime_error("Expected 1D array with 3 entries");
    }
    auto buf = arr.cast<py::array_t<double>>().unchecked<1>();
    return {buf(0), buf(1), buf(2)};
}

static std::array<double, 3> mat_vec_mul(const py::array_t<double>& mat, const std::array<double, 3>& vec) {
    auto m = mat.unchecked<2>();
    std::array<double, 3> res{};
    for (int i = 0; i < 3; ++i) {
        res[i] = m(i, 0) * vec[0] + m(i, 1) * vec[1] + m(i, 2) * vec[2];
    }
    return res;
}

static std::array<double, 3> vec_sub(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

static std::array<double, 3> vec_scale(const std::array<double, 3>& a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}

static std::array<double, 3> cross(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

static py::array_t<double> compute_damping_matrix(const py::array_t<double>& mass_matrix,
                                                  const py::array_t<double>& stiffness_matrix,
                                                  double mass_factor,
                                                  double stiffness_factor) {
    auto shape = mass_matrix.shape();
    py::array_t<double> result({shape[0], shape[1]});
    auto m = mass_matrix.unchecked<2>();
    auto k = stiffness_matrix.unchecked<2>();
    auto d = result.mutable_unchecked<2>();
    for (ssize_t i = 0; i < shape[0]; ++i) {
        for (ssize_t j = 0; j < shape[1]; ++j) {
            d(i, j) = mass_factor * m(i, j) + stiffness_factor * k(i, j);
        }
    }
    return result;
}

struct BeamData {
    py::object npz_data;
    py::object aux_data;
    py::array mode_basis;
    py::array mass_matrix;
    py::array stiffness_matrix;
    py::array damping_matrix;
    py::array reference_positions;
    py::array trig_list;
    py::array mPsiTildePsi;
    py::array mPsiTildePsiTilde;
    py::array mPhitTPsi;
    py::array mPhitTPsiTilde;
    py::array mXRefTildePsi;
    py::array mXRefTildePsiTilde;
    py::array root_mean;
    py::array tip_mean;
    py::array nodes_root;
    py::array nodes_tip;
    py::array weights_root;
    py::array weights_tip;
    std::array<double, 3> root_mean_vec;
    std::array<double, 3> tip_mean_vec;
    ssize_t num_nodes;
};

static BeamData load_beam(const std::filesystem::path& data_dir,
                          const std::string& prefix,
                          const FlexibleLinkParameters& params,
                          const py::module& numpy) {
    BeamData beam;
    std::string npz_file = (data_dir / (prefix + ".npz")).string();
    std::string aux_file = (data_dir / (prefix + "_aux.npz")).string();
    beam.npz_data = numpy.attr("load")(npz_file, py::arg("allow_pickle") = true);
    beam.aux_data = numpy.attr("load")(aux_file, py::arg("allow_pickle") = true);

    auto get_npz_array = [](const py::object& npz, const char* key) {
        return npz.attr("__getitem__")(py::str(key)).cast<py::array>();
    };

    beam.mode_basis = get_npz_array(beam.npz_data, "modeBasis");
    beam.mass_matrix = get_npz_array(beam.npz_data, "massMatrixReduced");
    beam.stiffness_matrix = get_npz_array(beam.npz_data, "stiffnessMatrixReduced");
    beam.reference_positions = get_npz_array(beam.npz_data, "xRef");
    beam.trig_list = get_npz_array(beam.npz_data, "trigList");
    beam.mPsiTildePsi = get_npz_array(beam.npz_data, "mPsiTildePsi");
    beam.mPsiTildePsiTilde = get_npz_array(beam.npz_data, "mPsiTildePsiTilde");
    beam.mPhitTPsi = get_npz_array(beam.npz_data, "mPhitTPsi");
    beam.mPhitTPsiTilde = get_npz_array(beam.npz_data, "mPhitTPsiTilde");
    beam.mXRefTildePsi = get_npz_array(beam.npz_data, "mXRefTildePsi");
    beam.mXRefTildePsiTilde = get_npz_array(beam.npz_data, "mXRefTildePsiTilde");

    beam.root_mean = get_npz_array(beam.aux_data, "root_mean");
    beam.tip_mean = get_npz_array(beam.aux_data, "tip_mean");
    beam.nodes_root = get_npz_array(beam.aux_data, "nodes_root");
    beam.nodes_tip = get_npz_array(beam.aux_data, "nodes_tip");
    beam.weights_root = get_npz_array(beam.aux_data, "weights_root");
    beam.weights_tip = get_npz_array(beam.aux_data, "weights_tip");
    beam.root_mean_vec = to_vector3(beam.root_mean);
    beam.tip_mean_vec = to_vector3(beam.tip_mean);

    beam.num_nodes = get_npz_array(beam.aux_data, "nodes").shape(0);

    beam.damping_matrix = compute_damping_matrix(
        beam.mass_matrix.cast<py::array_t<double>>(),
        beam.stiffness_matrix.cast<py::array_t<double>>(),
        params.mass_proportional_damping,
        params.stiffness_proportional_damping);

    return beam;
}

static py::list to_pylist(const std::vector<double>& values) {
    py::list lst;
    for (double v : values) {
        lst.append(v);
    }
    return lst;
}

static std::vector<py::ssize_t> array_to_index_vector(const py::array& arr) {
    auto buf = arr.cast<py::array_t<py::ssize_t>>().unchecked<1>();
    std::vector<py::ssize_t> result(buf.shape(0));
    for (ssize_t i = 0; i < buf.shape(0); ++i) {
        result[static_cast<size_t>(i)] = buf(i);
    }
    return result;
}

static std::vector<double> array_to_vector(const py::array& arr) {
    auto buf = arr.cast<py::array_t<double>>().unchecked<1>();
    std::vector<double> result(buf.shape(0));
    for (ssize_t i = 0; i < buf.shape(0); ++i) {
        result[static_cast<size_t>(i)] = buf(i);
    }
    return result;
}

static py::list build_reference_list(const std::array<double, 3>& position,
                                     const py::array_t<double>& rotation_parameters) {
    std::vector<double> values = {position[0], position[1], position[2]};
    auto rot = rotation_parameters.unchecked<1>();
    for (ssize_t i = 0; i < rot.shape(0); ++i) {
        values.push_back(rot(i));
    }
    return to_pylist(values);
}

static py::list build_velocity_list(const std::array<double, 3>& velocity,
                                    const py::array_t<double>& rotation_rate) {
    std::vector<double> values = {velocity[0], velocity[1], velocity[2]};
    auto rot = rotation_rate.unchecked<1>();
    for (ssize_t i = 0; i < rot.shape(0); ++i) {
        values.push_back(rot(i));
    }
    return to_pylist(values);
}

int main(int argc, char** argv) try {
    py::scoped_interpreter guard{};

    const FlexibleLinkParameters params;
    const InitialState init;
    const SimulationSetup sim_cfg;

    py::module exu = py::module::import("exudyn");
    py::module eii = py::module::import("exudyn.itemInterface");
    py::module numpy = py::module::import("numpy");
    py::module utilities = py::module::import("exudyn.utilities");

    std::filesystem::path script_dir = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path data_dir = script_dir / "data";

    std::cout << "Loading beam caches..." << std::endl;
    BeamData beam_first = load_beam(data_dir, "beam_first", params, numpy);
    BeamData beam_second = load_beam(data_dir, "beam_second", params, numpy);

    std::cout << "Beam data loaded." << std::endl;

    std::array<double, 3> omega1{0.0, 0.0, init.omega1};
    double omega2_total = init.omega1 + init.omega2;
    std::array<double, 3> omega2{0.0, 0.0, omega2_total};

    double psi1 = init.theta1 - 0.5 * M_PI;
    double psi2 = init.theta1 + init.theta2 - 0.5 * M_PI;

    py::array_t<double> rot1 = rotation_z(psi1);
    py::array_t<double> rot2 = rotation_z(psi2);

    auto hinge_vector_local = vec_sub(beam_first.tip_mean_vec, beam_first.root_mean_vec);
    std::array<double, 3> hinge_pos = mat_vec_mul(rot1, hinge_vector_local);
    std::array<double, 3> hinge_vel = cross(omega1, hinge_pos);

    std::array<double, 3> pos_first = vec_scale(mat_vec_mul(rot1, beam_first.root_mean_vec), -1.0);
    std::array<double, 3> rot1_root = mat_vec_mul(rot1, beam_first.root_mean_vec);
    std::array<double, 3> vel_first = vec_scale(cross(omega1, rot1_root), -1.0);

    std::array<double, 3> rot2_root = mat_vec_mul(rot2, beam_second.root_mean_vec);
    std::array<double, 3> pos_second = vec_sub(hinge_pos, rot2_root);
    std::array<double, 3> vel_second = vec_sub(hinge_vel, cross(omega2, rot2_root));

    py::object rot1_params = utilities.attr("RotationMatrix2RotXYZ")(rot1);
    py::object rot2_params = utilities.attr("RotationMatrix2RotXYZ")(rot2);

    py::object rot1_rates = utilities.attr("AngularVelocity2RotXYZ_t")(py::make_tuple(omega1[0], omega1[1], omega1[2]), rot1_params);
    py::object rot2_rates = utilities.attr("AngularVelocity2RotXYZ_t")(py::make_tuple(omega2[0], omega2[1], omega2[2]), rot2_params);

    py::object SC = exu.attr("SystemContainer")();
    py::object mbs = SC.attr("AddSystem")();

    std::cout << "System container initialized." << std::endl;

    py::object ObjectGround = eii.attr("ObjectGround");
    py::object MarkerBodyRigid = eii.attr("MarkerBodyRigid");
    py::object MarkerSuperElementRigid = eii.attr("MarkerSuperElementRigid");
    py::object VMarkerBodyRigid = eii.attr("VMarkerBodyRigid");
    py::object VMarkerSuperElementRigid = eii.attr("VMarkerSuperElementRigid");
    py::object GenericJoint = eii.attr("GenericJoint");
    py::object VGenericJoint = eii.attr("VGenericJoint");
    py::object SensorMarker = eii.attr("SensorMarker");
    py::object SensorSuperElement = eii.attr("SensorSuperElement");
    py::object MatrixContainer = exu.attr("MatrixContainer");
    py::object VObjectFFRF = eii.attr("VObjectFFRF");
    py::object ObjectFFRFreducedOrder = eii.attr("ObjectFFRFreducedOrder");
    py::object NodeRigidBodyRxyz = eii.attr("NodeRigidBodyRxyz");
    py::object NodeGenericODE2 = eii.attr("NodeGenericODE2");
    py::object LoadMassProportional = eii.attr("LoadMassProportional");

    py::object ground = mbs.attr("AddObject")(ObjectGround(py::arg("referencePosition") = py::make_tuple(0.0, 0.0, 0.0)));
    std::cout << "Ground object index: " << py::int_(ground).cast<int>() << std::endl;

    auto make_matrix_container = [&](const py::array& arr) {
        py::object mc = MatrixContainer();
        mc.attr("SetWithDenseMatrix")(arr, py::arg("useDenseMatrix") = false);
        return mc;
    };

    auto create_ffrf_body = [&](const BeamData& beam,
                                const std::array<double, 3>& position,
                                const std::array<double, 3>& velocity,
                                const std::array<double, 3>& omega,
                                const py::array_t<double>& rotation_params,
                                const py::array_t<double>& rotation_rates,
                                const py::array_t<double>& rotation_matrix,
                                const std::array<double, 3>& gravity,
                                const std::array<double, 4>& color,
                                py::object& node_rigid_out,
                                py::object& node_generic_out,
                                py::object& object_out) {
        py::list ref_coords = build_reference_list(position, rotation_params);
        py::list vel_coords = build_velocity_list(velocity, rotation_rates);

        node_rigid_out = mbs.attr("AddNode")(NodeRigidBodyRxyz(py::arg("referenceCoordinates") = ref_coords,
                                                                py::arg("initialVelocities") = vel_coords));

        ssize_t n_modes = beam.mode_basis.shape(1);
        node_generic_out = mbs.attr("AddNode")(NodeGenericODE2(py::arg("numberOfODE2Coordinates") = n_modes,
                                                                py::arg("referenceCoordinates") = std::vector<double>(static_cast<size_t>(n_modes), 0.0),
                                                                py::arg("initialCoordinates") = std::vector<double>(static_cast<size_t>(n_modes), 0.0),
                                                                py::arg("initialCoordinates_t") = std::vector<double>(static_cast<size_t>(n_modes), 0.0)));

        py::object mass_mc = make_matrix_container(beam.mass_matrix);
        py::object stiffness_mc = make_matrix_container(beam.stiffness_matrix);
        py::object damping_mc = make_matrix_container(beam.damping_matrix);

        py::object visualization = VObjectFFRF(py::arg("triangleMesh") = beam.trig_list,
                                               py::arg("color") = color,
                                               py::arg("showNodes") = true);

        std::vector<double> gravity_vec = {gravity[0], gravity[1], gravity[2]};

        py::list node_numbers;
        node_numbers.append(node_rigid_out);
        node_numbers.append(node_generic_out);

        object_out = mbs.attr("AddObject")(ObjectFFRFreducedOrder(py::arg("nodeNumbers") = node_numbers,
                                                                     py::arg("massMatrixReduced") = mass_mc,
                                                                     py::arg("stiffnessMatrixReduced") = stiffness_mc,
                                                                     py::arg("dampingMatrixReduced") = damping_mc,
                                                                     py::arg("modeBasis") = beam.mode_basis,
                                                                     py::arg("referencePositions") = beam.reference_positions,
                                                                     py::arg("physicsMass") = beam.npz_data.attr("__getitem__")(py::str("totalMass")),
                                                                     py::arg("physicsInertia") = beam.npz_data.attr("__getitem__")(py::str("inertiaLocal")),
                                                                     py::arg("physicsCenterOfMass") = beam.npz_data.attr("__getitem__")(py::str("chiU")),
                                                                     py::arg("mPsiTildePsi") = beam.mPsiTildePsi,
                                                                     py::arg("mPsiTildePsiTilde") = beam.mPsiTildePsiTilde,
                                                                     py::arg("mPhitTPsi") = beam.mPhitTPsi,
                                                                     py::arg("mPhitTPsiTilde") = beam.mPhitTPsiTilde,
                                                                     py::arg("mXRefTildePsi") = beam.mXRefTildePsi,
                                                                     py::arg("mXRefTildePsiTilde") = beam.mXRefTildePsiTilde,
                                                                     py::arg("visualization") = visualization,
                                                                     py::arg("computeFFRFterms") = true));

        if (gravity_vec[0] != 0.0 || gravity_vec[1] != 0.0 || gravity_vec[2] != 0.0) {
        py::object mass_marker = mbs.attr("AddMarker")(eii.attr("MarkerBodyMass")(py::arg("bodyNumber") = object_out));
        mbs.attr("AddLoad")(LoadMassProportional(py::arg("markerNumber") = mass_marker,
                                                       py::arg("loadVector") = gravity_vec));
        }
    };

    std::cout << "Creating first body..." << std::endl;
    py::object node_rigid_1, node_generic_1, object_1;
    create_ffrf_body(beam_first,
                     pos_first,
                     vel_first,
                     omega1,
                     rot1_params.cast<py::array_t<double>>(),
                     rot1_rates.cast<py::array_t<double>>(),
                     rot1,
                     {0.0, -params.gravity, 0.0},
                     {0.1, 0.6, 0.9, 1.0},
                     node_rigid_1,
                     node_generic_1,
                     object_1);

    std::cout << "First body created. Creating second body..." << std::endl;
    py::object node_rigid_2, node_generic_2, object_2;
    create_ffrf_body(beam_second,
                     pos_second,
                     vel_second,
                     omega2,
                     rot2_params.cast<py::array_t<double>>(),
                     rot2_rates.cast<py::array_t<double>>(),
                     rot2,
                     {0.0, -params.gravity, 0.0},
                     {0.9, 0.4, 0.2, 1.0},
                     node_rigid_2,
                     node_generic_2,
                     object_2);

    std::cout << "Bodies created. Creating joints and sensors..." << std::endl;

    std::cout << "Adding ground marker..." << std::endl;
    py::object marker_ground = mbs.attr("AddMarker")(MarkerBodyRigid(py::arg("bodyNumber") = ground,
                                                                      py::arg("localPosition") = std::vector<double>{0.0, 0.0, 0.0},
                                                                      py::arg("visualization") = VMarkerBodyRigid(py::arg("show") = false)));

    auto create_super_marker = [&](const py::object& body,
                                   const py::array& nodes,
                                   const py::array& weights) {
        std::cout << "Creating super-element marker with " << nodes.shape(0) << " nodes" << std::endl;
        std::vector<py::ssize_t> node_vec = array_to_index_vector(nodes);
        std::vector<double> weight_vec = array_to_vector(weights);
        return mbs.attr("AddMarker")(MarkerSuperElementRigid(py::arg("bodyNumber") = body,
                                                               py::arg("meshNodeNumbers") = node_vec,
                                                               py::arg("weightingFactors") = weight_vec,
                                                               py::arg("useAlternativeApproach") = true,
                                                               py::arg("visualization") = VMarkerSuperElementRigid(py::arg("show") = true)));
    };

    std::cout << "Adding markers for first body..." << std::endl;
    py::object marker_first_root = create_super_marker(object_1, beam_first.nodes_root, beam_first.weights_root);
    py::object marker_first_tip = create_super_marker(object_1, beam_first.nodes_tip, beam_first.weights_tip);
    std::cout << "Adding markers for second body..." << std::endl;
    py::object marker_second_root = create_super_marker(object_2, beam_second.nodes_root, beam_second.weights_root);
    py::object marker_second_tip = create_super_marker(object_2, beam_second.nodes_tip, beam_second.weights_tip);

    std::cout << "Adding joints..." << std::endl;
    std::vector<int> joint_axes = {1, 1, 1, 1, 1, 0};
    py::list joint_markers_ground;
    joint_markers_ground.append(marker_ground);
    joint_markers_ground.append(marker_first_root);
    mbs.attr("AddObject")(GenericJoint(py::arg("markerNumbers") = joint_markers_ground,
                                        py::arg("constrainedAxes") = joint_axes,
                                        py::arg("visualization") = VGenericJoint(py::arg("show") = false)));

    py::list joint_markers_links;
    joint_markers_links.append(marker_first_tip);
    joint_markers_links.append(marker_second_root);
    mbs.attr("AddObject")(GenericJoint(py::arg("markerNumbers") = joint_markers_links,
                                        py::arg("constrainedAxes") = joint_axes,
                                        py::arg("visualization") = VGenericJoint(py::arg("show") = false)));

    std::cout << "Registering tip sensor..." << std::endl;
    py::object tip_sensor = mbs.attr("AddSensor")(SensorMarker(py::arg("markerNumber") = marker_second_tip,
                                                                py::arg("outputVariableType") = exu.attr("OutputVariableType").attr("Position"),
                                                                py::arg("storeInternal") = true));

    std::vector<py::object> sensors_beam1;
    std::vector<py::object> sensors_beam2;

    auto register_node_sensors = [&](const py::object& body,
                                     ssize_t node_count,
                                     std::vector<py::object>& container) {
        for (ssize_t node_idx = 0; node_idx < node_count; ++node_idx) {
            py::object sensor = mbs.attr("AddSensor")(SensorSuperElement(py::arg("bodyNumber") = body,
                                                                           py::arg("meshNodeNumber") = node_idx,
                                                                           py::arg("outputVariableType") = exu.attr("OutputVariableType").attr("Position"),
                                                                           py::arg("storeInternal") = true));
            container.push_back(sensor);
        }
    };

    if (sim_cfg.store_trajectory) {
        register_node_sensors(object_1, beam_first.num_nodes, sensors_beam1);
        register_node_sensors(object_2, beam_second.num_nodes, sensors_beam2);
    }

    std::cout << "Assembling model..." << std::endl;
    mbs.attr("Assemble")();

    std::cout << "Model assembled. Configuring solver..." << std::endl;

    py::object settings = exu.attr("SimulationSettings")();
    py::object timeIntegration = settings.attr("timeIntegration");
    py::object newtonSettings = timeIntegration.attr("newton");
    py::object generalAlpha = timeIntegration.attr("generalizedAlpha");

    std::cout << "Setting solver end time..." << std::endl;
    py::setattr(timeIntegration, "endTime", py::cast(sim_cfg.end_time));
    std::cout << "Setting number of steps..." << std::endl;
    int number_of_steps = static_cast<int>(sim_cfg.end_time / sim_cfg.step_size);
    py::setattr(timeIntegration, "numberOfSteps", py::cast(number_of_steps));
    std::cout << "Enabling adaptive step..." << std::endl;
    py::setattr(timeIntegration, "adaptiveStep", py::cast(true));
    std::cout << "Adaptive step set." << std::endl;
    py::setattr(timeIntegration, "automaticStepSize", py::cast(false));
    std::cout << "Automatic step size disabled." << std::endl;
    py::setattr(timeIntegration, "startTime", py::cast(0.0));
    std::cout << "Start time set." << std::endl;
    py::setattr(timeIntegration, "stepInformation", py::cast(0));
    std::cout << "Step info set." << std::endl;
    py::setattr(timeIntegration, "verboseMode", py::cast(1));
    std::cout << "Verbose mode set." << std::endl;
    py::setattr(timeIntegration, "verboseModeFile", py::cast(0));
    std::cout << "Verbose mode file set." << std::endl;

    py::setattr(generalAlpha, "spectralRadius", py::cast(0.85));
    std::cout << "Spectral radius set." << std::endl;

    std::cout << "Configuring Newton settings..." << std::endl;
    py::setattr(newtonSettings, "useModifiedNewton", py::bool_(true));

    std::cout << "Selecting linear solver..." << std::endl;
    py::setattr(settings, "linearSolverType", exu.attr("LinearSolverType").attr("EigenSparse"));
    py::setattr(settings, "displayStatistics", py::bool_(true));

    double sensors_period = sim_cfg.store_trajectory ? 1.0 / static_cast<double>(sim_cfg.frames_per_second)
                                                     : sim_cfg.step_size;
    std::cout << "Configuring sensor write period..." << std::endl;
    py::object solutionSettings = settings.attr("solutionSettings");
    py::setattr(solutionSettings, "sensorsWritePeriod", py::cast(sensors_period));
    py::setattr(solutionSettings, "writeSolutionToFile", py::cast(false));
    std::cout << "Solver configuration finished." << std::endl;

    std::cout << "Calling SolveDynamic..." << std::endl;
    py::object solverType = exu.attr("DynamicSolverType").attr("TrapezoidalIndex2");
    py::object success = mbs.attr("SolveDynamic")(settings,
                                                   py::arg("solverType") = solverType,
                                                   py::arg("storeSolver") = false,
                                                   py::arg("showHints") = false,
                                                   py::arg("showCausingItems") = false,
                                                   py::arg("autoAssemble") = false);

    bool success_flag = success.cast<bool>();
    std::cout << "Simulation success: " << std::boolalpha << success_flag << '\n';

    std::cout << "Collecting responses..." << std::endl;

    py::object tip_data = mbs.attr("GetSensorStoredData")(tip_sensor);

    auto gather_sensor_positions = [&](const std::vector<py::object>& sensors) -> py::array {
        if (sensors.empty()) {
            return py::array();
        }

        py::array_t<double> first_data = mbs.attr("GetSensorStoredData")(sensors.front()).cast<py::array_t<double>>();
        auto first_buf = first_data.unchecked<2>();
        ssize_t steps = first_buf.shape(0);
        ssize_t dim = first_buf.shape(1) - 1;  // skip time column
        ssize_t sensor_count = static_cast<ssize_t>(sensors.size());

        py::array_t<double> positions({steps, sensor_count, dim});
        auto pos_buf = positions.mutable_unchecked<3>();

        for (ssize_t step = 0; step < steps; ++step) {
            for (ssize_t d = 0; d < dim; ++d) {
                pos_buf(step, 0, d) = first_buf(step, d + 1);
            }
        }

        for (ssize_t sensor_idx = 1; sensor_idx < sensor_count; ++sensor_idx) {
            py::array_t<double> sensor_data = mbs.attr("GetSensorStoredData")(sensors[static_cast<size_t>(sensor_idx)]).cast<py::array_t<double>>();
            auto sensor_buf = sensor_data.unchecked<2>();
            if (sensor_buf.shape(0) != steps || sensor_buf.shape(1) != dim + 1) {
                throw std::runtime_error("Sensor data shape mismatch when collecting positions");
            }
            for (ssize_t step = 0; step < steps; ++step) {
                for (ssize_t d = 0; d < dim; ++d) {
                    pos_buf(step, sensor_idx, d) = sensor_buf(step, d + 1);
                }
            }
        }

        return positions;
    };

    py::array beam1_positions = gather_sensor_positions(sensors_beam1);
    py::array beam2_positions = gather_sensor_positions(sensors_beam2);

    py::array_t<double> tip_array = tip_data.cast<py::array_t<double>>();
    py::array_t<double> time_array(py::array::ShapeContainer{tip_array.shape(0)});
    auto tip_buf = tip_array.unchecked<2>();
    auto time_buf = time_array.mutable_unchecked<1>();
    for (ssize_t i = 0; i < tip_array.shape(0); ++i) {
        time_buf(i) = tip_buf(i, 0);
    }

    py::object python_response = numpy.attr("load")((data_dir / "python_response.npz").string(), py::arg("allow_pickle") = true);
    py::array tip_python = python_response.attr("__getitem__")(py::str("tip_sensor")).cast<py::array>();
    py::array tip_cpp = tip_data.cast<py::array>();

    py::object diff = numpy.attr("abs")(tip_cpp - tip_python);
    double max_difference = diff.attr("max")().cast<double>();
    std::cout << "Maximum tip position difference compared to Python: " << max_difference << '\n';

    numpy.attr("savez")((data_dir / "cpp_response.npz").string(),
                         py::arg("tip_sensor") = tip_cpp,
                         py::arg("beam1_positions") = beam1_positions,
                         py::arg("beam2_positions") = beam2_positions,
                         py::arg("time") = time_array);

    std::cout << "Stored C++ response to " << (data_dir / "cpp_response.npz") << '\n';

    return 0;
} catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
}
