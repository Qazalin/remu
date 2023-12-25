
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_2925n21>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001718: 86039F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000171C: BF870009
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001720: 84808202
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s2, s6, s0                                       // 000000001728: 80020006
	s_addc_u32 s3, s7, s1                                      // 00000000172C: 82030107
	s_add_u32 s6, s8, s0                                       // 000000001730: 80060008
	s_addc_u32 s7, s9, s1                                      // 000000001734: 82070109
	s_load_b32 s2, s[2:3], null                                // 000000001738: F4000081 F8000000
	s_load_b32 s3, s[6:7], null                                // 000000001740: F40000C3 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001748: BF89FC07
	v_cmp_lt_f32_e64 s6, s2, s3                                // 00000000174C: D4110006 00000602
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001754: BF870001
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001758: 8B6A067E
	s_cbranch_vccnz 10                                         // 00000000175C: BFA4000A <E_2925n21+0x88>
	v_cmp_lt_f32_e64 s6, s3, s2                                // 000000001760: D4110006 00000403
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001768: BF870001
	s_and_b32 vcc_lo, exec_lo, s6                              // 00000000176C: 8B6A067E
	s_cbranch_vccnz 7                                          // 000000001770: BFA40007 <E_2925n21+0x90>
	v_add_f32_e64 v0, s2, s3                                   // 000000001774: D5030000 00000602
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000177C: BF870001
	v_mul_f32_e32 v0, 0.5, v0                                  // 000000001780: 100000F0
	s_branch 3                                                 // 000000001784: BFA00003 <E_2925n21+0x94>
	v_mov_b32_e32 v0, s3                                       // 000000001788: 7E000203
	s_branch 1                                                 // 00000000178C: BFA00001 <E_2925n21+0x94>
	v_mov_b32_e32 v0, s2                                       // 000000001790: 7E000202
	v_mov_b32_e32 v1, 0                                        // 000000001794: 7E020280
	s_add_u32 s0, s4, s0                                       // 000000001798: 80000004
	s_addc_u32 s1, s5, s1                                      // 00000000179C: 82010105
	global_store_b32 v1, v0, s[0:1]                            // 0000000017A0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017A8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017AC: BFB60003
	s_endpgm                                                   // 0000000017B0: BFB00000
