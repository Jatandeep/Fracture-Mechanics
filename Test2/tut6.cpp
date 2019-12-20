#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>

using namespace dealii;

template<int dim>
class Step6{
public:
    Step6();
    ~Step6();
    void run();
private:

    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    ConstraintMatrix constraints;

    Vector<double> solution;
    Vector<double> system_rhs;

};

template<int dim>
Step6<dim>::~Step6(){
    system_matrix.clear();
}

template<int dim>
double coefficient(const Point<dim> &p){
    if(p.square()<0.5*0.5)
        return 20;
    else
        return 1;
}

template<int dim>
Step6<dim>::Step6():fe(2),dof_handler(triangulation){}

template<int dim>
void Step6<dim>::setup_system(){

    dof_handler.distribute_dofs(fe);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,constraints);
    VectorTools::interpolate_boundary_values(dof_handler,0,Functions::ZeroFunction<dim>(),constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,dsp,constraints,false);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);


}

template<int dim>
void Step6<dim>::assemble_system(){

    const QGauss<dim> quadrature_formula(3);
    FEValues<dim> fe_values(fe,quadrature_formula,update_values|update_gradients|update_quadrature_points|update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(),endc=dof_handler.end();

    for(;cell!=endc;++cell){
        fe_values.reinit(cell);

        cell_matrix=0;
        cell_rhs=0;

        for(unsigned int q_index=0;q_index<n_q_points;++q_index){
            const double current_coefficient = coefficient<dim>(fe_values.quadrature_point(q_index));
            for(unsigned int i=0;i<dofs_per_cell;++i)
            {
                for(unsigned int j=0;j<dofs_per_cell;++j)
                    cell_matrix(i,j) +=(current_coefficient*fe_values.shape_grad(i,q_index)*fe_values.shape_grad(j,q_index)*fe_values.JxW(q_index));

                    cell_rhs(i)+=(fe_values.shape_value(i,q_index))*1.0*fe_values.JxW(q_index);
            }
        }
        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(cell_matrix,cell_rhs,local_dof_indices,system_matrix,system_rhs);

    }

}

template<int dim>
void Step6<dim>::solve(){
    SolverControl solver_control (1000,1e-12);
    SolverCG<> solver (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix,1.2);

    solver.solve(system_matrix,solution,system_rhs,preconditioner);

    constraints.distribute(solution);
}

template<int dim>
void Step6<dim>::refine_grid(){
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,QGauss<dim-1>(3),typename FunctionMap<dim>::type(),solution,estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,estimated_error_per_cell,0.3,0.03);

    triangulation.execute_coarsening_and_refinement();

}

template<int dim>
void Step6<dim>::output_results(const unsigned int cycle) const{
    {
        GridOut grid_out;
        std::ofstream output("gridtut6-"+std::to_string(cycle)+".eps");
        grid_out.write_eps(triangulation,output);
    }
    {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,"solution");
    data_out.build_patches();

    std::ofstream output("solutiontut6-"+std::to_string(cycle)+".vtk");
    data_out.write_vtk(output);
    }
}

template<int dim>
void Step6<dim>::run(){

  for(unsigned int cycle=0;cycle<8;++cycle){
        std::cout<<"Cycle "<<cycle<<":"<<std::endl;
        if(cycle==0)
        {
            GridGenerator::hyper_ball(triangulation);
            triangulation.refine_global(1);
        }
        else
            refine_grid();

        std::cout << "   Number of active cells: "
                  << triangulation.n_active_cells()
                  << std::endl;

            setup_system();

        std::cout << "   Total number of cells: "
                  << triangulation.n_cells()
                  << std::endl;

                    assemble_system();
                    solve();
                    output_results(cycle);
    }
}

int main()
{
    try {
        Step6<2> laplace_problem;
        laplace_problem.run();
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }


    return 0;
}
