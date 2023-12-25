
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_2925n22>:
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
	s_load_b32 s3, s[2:3], null                                // 000000001738: F40000C1 F8000000
	s_load_b32 s2, s[6:7], null                                // 000000001740: F4000083 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001748: BF89FC07
	v_cmp_ngt_f32_e64 s6, s3, s2                               // 00000000174C: D41B0006 00000403
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001754: BF870001
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001758: 8B6A067E
	s_cbranch_vccz 14                                          // 00000000175C: BFA3000E <E_2925n22+0x98>
	v_cmp_gt_f32_e64 s6, s2, s3                                // 000000001760: D4140006 00000602
	s_xor_b32 s3, s3, 0x80000000                               // 000000001768: 8D03FF03 80000000
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001770: BF870119
	v_mov_b32_e32 v0, s3                                       // 000000001774: 7E000203
	s_and_b32 vcc_lo, exec_lo, s6                              // 000000001778: 8B6A067E
	s_cbranch_vccnz 4                                          // 00000000177C: BFA40004 <E_2925n22+0x90>
	v_sub_f32_e64 v0, s3, s2                                   // 000000001780: D5040000 00000403
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001788: BF870001
	v_mul_f32_e32 v0, 0.5, v0                                  // 00000000178C: 100000F0
	s_cbranch_execz 1                                          // 000000001790: BFA50001 <E_2925n22+0x98>
	s_branch 4                                                 // 000000001794: BFA00004 <E_2925n22+0xa8>
	s_xor_b32 s2, s2, 0x80000000                               // 000000001798: 8D02FF02 80000000
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017A0: BF870009
	v_mov_b32_e32 v0, s2                                       // 0000000017A4: 7E000202
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017A8: BF870001
	v_xor_b32_e32 v0, 0x80000000, v0                           // 0000000017AC: 3A0000FF 80000000
	v_mov_b32_e32 v1, 0                                        // 0000000017B4: 7E020280
	s_add_u32 s0, s4, s0                                       // 0000000017B8: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017BC: 82010105
	global_store_b32 v1, v0, s[0:1]                            // 0000000017C0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017C8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017CC: BFB60003
	s_endpgm                                                   // 0000000017D0: BFB00000
