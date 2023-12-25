
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_45_65n1>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s3, s15, 0x41                                    // 000000001714: 9603FF0F 00000041
	s_mov_b32 s2, s15                                          // 00000000171C: BE82000F
	s_add_i32 s8, s3, s14                                      // 000000001720: 81080E03
	v_mov_b32_e32 v1, 0                                        // 000000001724: 7E020280
	s_ashr_i32 s9, s8, 31                                      // 000000001728: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000172C: BF8704D9
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001730: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001734: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001738: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000173C: 82070907
	s_ashr_i32 s3, s15, 31                                     // 000000001740: 86039F0F
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001744: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001748: BF870009
	s_add_u32 s0, s0, s2                                       // 00000000174C: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001750: 82010301
	s_load_b32 s2, s[6:7], null                                // 000000001754: F4000083 F8000000
	s_load_b32 s0, s[0:1], null                                // 00000000175C: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	v_add_f32_e64 v0, s2, s0                                   // 000000001768: D5030000 00000002
	s_add_u32 s0, s4, s8                                       // 000000001770: 80000804
	s_addc_u32 s1, s5, s9                                      // 000000001774: 82010905
	global_store_b32 v1, v0, s[0:1]                            // 000000001778: DC6A0000 00000001
	s_nop 0                                                    // 000000001780: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001784: BFB60003
	s_endpgm                                                   // 000000001788: BFB00000
