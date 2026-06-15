using Oceananigans
using Oceananigans.Units

using CairoMakie
using CUDA
using Printf
using Random
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

# ──────────────────────────────────────────────────────────────────────
# 1. TIME-VARYING PHYSICAL BOUNDARY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

# Zonal Wind Stress (τx) - Handles Tipping Points & Distractors
@inline function wind_stress_forcing(x, y, t, p)
    τ₀ = p.tau_base         
    τ_current = τ₀
    t_years = t / 365days 

    # --- POSITIVE CASES (Bifurcations) ---
    if p.scenario == 1      # BIFURCATION_COLLAPSE
        ramp = 1.0 / (1.0 + exp(-(t_years - 20.0) / 2.0))
        τ_current = τ₀ * (1.0 - 0.8 * ramp) 
        
    elseif p.scenario == 2  # VARIANCE (Critical Slowing Down precursor)
        ramp = 1.0 / (1.0 + exp(-(t_years - 20.0) / 4.0))
        # Drop mean slightly, inject massive high-frequency noise variation
        τ_current = τ₀ * (1.0 - 0.3 * ramp) + (0.01 + 0.08 * ramp) * randn()
        
    elseif p.scenario == 3  # HOPF (Emergent Limit Cycle / Oscillation)
        ramp = 1.0 / (1.0 + exp(-(t_years - 20.0) / 2.0))
        amplitude = 0.15 * ramp
        τ_current = τ₀ + amplitude * sin(2π * t_years / 0.5) # 6-month cycle growth

    # --- NEGATIVE CASES (Hard Distractors / Nulls) ---
    elseif p.scenario == 4  # STORM (Sudden transient shock)
        event_center = 15.0 
        if abs(t_years - event_center) < (10days / 365days)
            τ_current += 0.3 * randn() 
        end
        
    elseif p.scenario == 5  # EDDY (Temporary localized mean shift / blocking)
        event_center = 25.0
        if 0.0 < (t_years - event_center) < (90days / 365days)
            τ_current += 0.08 
        end
        
    elseif p.scenario == 6  # SEASONAL (Cyclical baseline variance)
        Base_seasonal = 0.03 * sin(2π * t_years / 1.0)
        τ_current += Base_seasonal
        
    elseif p.scenario == 7  # NEAR_MISS (Perilous drop that recovers)
        approach_c = 15.0 
        recovery_c = 25.0 
        near_miss_ramp = (
            1.0 / (1.0 + exp(-(t_years - approach_c) / 1.5)) - 
            1.0 / (1.0 + exp(-(t_years - recovery_c) / 1.5))
        )
        near_miss_ramp = clamp(near_miss_ramp, 0.0, 1.0)
        τ_current = τ₀ * (1.0 - 0.70 * near_miss_ramp) # Dips close to collapse but resets
    end
    # Smooth startup protection: Ramps up wind from 0 to full over the first 5 days
    startup_ramp = min(1.0, t / 5days)
    # Map back to the spatial double-gyre profile across the basin
    return -τ_current * cos(π * y / 4000kilometers) * startup_ramp
end

# Sea Surface Temperature Flux - Incorporates SSP370 Global Warming Trend
@inline function ssp370_temperature_flux(x, y, t, p)
    t_years = t / 365days
    # Background baseline differential solar heating
    base_flux = -1e-5 * cos(π * y / 4000kilometers) 
    # Continuous climate forcing amplification (Negative flux in Oceananigans injects heat)
    warming_anomaly = p.warming_rate_per_year * t_years
    startup_ramp = min(1.0, t / 5days)
    return base_flux - warming_anomaly * startup_ramp
end

# Salinity Surface Flux - Incorporates Hydrological Cycle Intensification (E - P)
@inline function ssp370_salinity_flux(x, y, t, p)
    t_years = t / 365days
    # Base Evaporation-Precipitation spatial footprint
    base_ep_flux = p.base_evap * sin(π * y / 4000kilometers) 
    # SSP370 forcing amplifies moisture transport (wet gets wetter, dry gets drier)
    ssp370_amplification = 1.0 + (p.salinity_trend_rate * t_years)
    startup_ramp = min(1.0, t / 5days)
    return base_ep_flux * ssp370_amplification * startup_ramp
end

# Free Surface Mass Flux - Models Accelerated Sea Level Rise (SLR Volume Input)
@inline function ssp370_slr_flux(x, y, t, p)
    t_years = t / 365days
    # Linear projection combined with an acceleration coefficient over time
    slr_velocity = p.base_slr_velocity + (p.slr_acceleration * t_years)
    startup_ramp = min(1.0, t / 5days)
    return slr_velocity * startup_ramp
end





# ──────────────────────────────────────────────────────────────────────
# 2. MAIN ENSEMBLE EXECUTION ENGINE
# ──────────────────────────────────────────────────────────────────────

function run_ensemble_member(scenario_id::Int, run_id::Int; use_gpu::Bool=false)
    @info "Initializing Simulation Setup for Scenario: $scenario_id, Run ID: $run_id"

    # ## The grid
    #
    # We use 128²×64 grid points with 1 m grid spacing in the horizontal and
    # varying spacing in the vertical, with higher resolution closer to the
    # surface. Here we use a stretching function for the vertical nodes that
    # maintains relatively constant vertical spacing in the mixed layer, which
    # is desirable from a numerical standpoint:
    
    Nx = Ny = 128    # number of points in each of horizontal directions
    Nz = 64          # number of points in the vertical direction
    
    Lx = Ly = 128    # (m) domain horizontal extents
    Lz = 64          # (m) domain depth
    
    refinement = 1.2 # controls spacing near surface (higher means finer spaced)
    stretching = 12  # controls rate of stretching at bottom
    
    ## Normalized height ranging from 0 to 1
    h(k) = (k - 1) / Nz
    
    ## Linear near-surface generator
    ζ₀(k) = 1 + (h(k) - 1) / refinement
    
    ## Bottom-intensified stretching function
    Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))
    
    ## Generating function
    z_interfaces(k) = Lz * (ζ₀(k) * Σ(k) - 1)
    
    grid = RectilinearGrid(GPU(),
                           size = (Nx, Nx, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = z_interfaces)
    
    # We plot vertical spacing versus depth to inspect the prescribed grid stretching:
    
    fig = Figure(size=(1200, 800))
    ax = Axis(fig[1, 1], ylabel = "z (m)", xlabel = "Vertical spacing (m)")
    
    lines!(ax, zspacings(grid, Center()))
    scatter!(ax, zspacings(grid, Center()))
    
    current_figure() #hide
    fig
    
    
    # ## Buoyancy that depends on temperature and salinity
    #
    # We use the `SeawaterBuoyancy` model with the TEOS10 equation of state,
    
    ρₒ = 1026 # kg m⁻³, average density at the surface of the world ocean
    equation_of_state = TEOS10EquationOfState(reference_density=ρₒ)
    buoyancy = SeawaterBuoyancy(; equation_of_state)

    Q = 200   # W m⁻², surface _heat_ flux
    cᴾ = 3991 # J K⁻¹ kg⁻¹, typical heat capacity for seawater
    
    Jᵀ = Q / (ρₒ * cᴾ) # K m s⁻¹, surface _temperature_ flux
    
    
    # Define a clean helper for uniform randomization
    randuniform(low, high) = low + (high - low) * rand()

    # Define Randomized Parameter Matrix (Domain Randomization Block)
    base_wind = 0.1 * randuniform(0.8, 1.2) # Baseline wind stress varies between 0.08 and 0.12
    
    p = (
        scenario = scenario_id,
        tau_base = base_wind/1025.0, #≈ 9.75e-5 m²/s² kinematic stress
        
        # SSP370 Climate Forcing Variables
        warming_rate_per_year = 5e-7 * randuniform(0.8, 1.2), 
        base_slr_velocity     = (0.015 / 365days) * randuniform(0.7, 1.3), # ~1.5cm base rise per year
        slr_acceleration      = 1e-11,
        base_evap             = 1e-5 * randuniform(0.9, 1.1),
        salinity_trend_rate   = 0.006 * randuniform(0.8, 1.2) # ~0.6% amplification/year
    )

    # Grid Construction: Idealized 4000km square flat-bottom basin
    arch = use_gpu ? GPU() : CPU()

    # Bind Functional Boundary Conditions with Parametric Arguments
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(wind_stress_forcing, parameters=p))
    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(ssp370_temperature_flux, parameters=p))
    S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(ssp370_salinity_flux, parameters=p))

    # # Calibrate Fluid Viscosity to Induce/Suppress Eddy Bistability
    # visc = (scenario_id == 5 || base_wind > 0.11) ? 1e2 : 4e3

    # Instantiate the Hydrostatic Fluid Core Model 
    model = HydrostaticFreeSurfaceModel(grid; buoyancy,
                                        advection = WENO(order=7),
                                        tracers = (:T, :S),
                                        boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

    ## Random noise damped at top and bottom
    Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
    
    ## Temperature initial condition: a stable density gradient with random noise superposed.
    Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 2e-6 * Ξ(z)
    
    ## Velocity initial condition: random noise scaled by the friction velocity.
    uᵢ(x, y, z) = sqrt(abs(τx)) * 1e-3 * Ξ(z)
    
    ## `set!` the `model` fields using functions or constants:
    set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

    # Simulation Scheduling Configuration
    simulation = Simulation(model, Δt=2minutes, stop_time=50 * 365days)
    
    conjure_time_step_wizard!(simulation, cfl=0.7)

    ## Print a progress message
    progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                    iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                    maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))
    
    add_callback!(simulation, progress_message, IterationInterval(200))

    ## Create a NamedTuple with eddy viscosity
    eddy_viscosity = (; νₑ = model.closure_fields.νₑ)
    # --- THE VIRTUAL MOORING LAYER ---
    # Downsample spatially during execution loop to extract high-frequency local signals
    # Coordinate index (30, 64) is positioned directly inside the high-shear boundary current jet.
    mooring_filename_3d = "mooring_scen$(scenario_id)_run$(run_id)_3d.jld2"
    mooring_filename_2d = "mooring_scen$(scenario_id)_run$(run_id)_2d.jld2"
    # 1. Writer for 3D Water Column Fields (u, v, T, S)
    simulation.output_writers[:mooring_3d] = JLD2Writer(
        model, 
        (
            u = model.velocities.u, 
            v = model.velocities.v, 
            T = model.tracers.T, 
            S = model.tracers.S
        ),
        indices = (30, 64, 1:4), # Extracts all 4 depth panels
        schedule = TimeInterval(1days),
        filename = mooring_filename_3d,
        overwrite_existing = true
    )

    # 2. Writer for 2D Surface Fields (Sea Surface Height / Displacement)
    simulation.output_writers[:mooring_2d] = JLD2Writer(
        model, 
        (
            eta = model.free_surface.displacement,
        ),
        indices = (30, 64, :), # ':' automatically grabs the correct single surface index slice safely
        schedule = TimeInterval(1days),
        filename = mooring_filename_2d,
        overwrite_existing = true
    )
    
    # Fire numerical calculation engine
    @info "Starting physics computation loop for 50 simulated years..."
    run!(simulation)
    @info "Simulation run successfully preserved to files: $mooring_filename_3d and $mooring_filename_2d"
end


# ──────────────────────────────────────────────────────────────────────
# 3. DIRECT EXECUTION TESTING BLOCK
# ──────────────────────────────────────────────────────────────────────
# You can uncomment this block to execute a quick test run locally.
# Scenario 7: Near Miss Scenario, Run ID: 1, Running on CPU

run_ensemble_member(7, 1, use_gpu=false)