
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_3_2>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001614: BE82000F
	s_cmp_gt_i32 s15, 2                                        // 000000001618: BF02820F
	s_mov_b32 s8, 0                                            // 00000000161C: BE880080
	s_cbranch_scc1 8                                           // 000000001620: BFA20008 <r_3_2+0x44>
	s_ashr_i32 s3, s2, 31                                      // 000000001624: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001628: BF870009
	s_lshl_b64 s[8:9], s[2:3], 2                               // 00000000162C: 84888202
	s_waitcnt lgkmcnt(0)                                       // 000000001630: BF89FC07
	s_add_u32 s8, s6, s8                                       // 000000001634: 80080806
	s_addc_u32 s9, s7, s9                                      // 000000001638: 82090907
	s_load_b32 s8, s[8:9], null                                // 00000000163C: F4000204 F8000000
	s_cmp_lt_i32 s2, 2                                         // 000000001644: BF048202
	s_cbranch_scc1 7                                           // 000000001648: BFA20007 <r_3_2+0x68>
	s_ashr_i32 s3, s2, 31                                      // 00000000164C: 86039F02
	s_mov_b32 s9, 0                                            // 000000001650: BE890080
	s_waitcnt lgkmcnt(0)                                       // 000000001654: BF89FC07
	s_load_b32 s10, s[0:1], null                               // 000000001658: F4000280 F8000000
	s_cbranch_execz 4                                          // 000000001660: BFA50004 <r_3_2+0x74>
	s_branch 10                                                // 000000001664: BFA0000A <r_3_2+0x90>
	s_waitcnt lgkmcnt(0)                                       // 000000001668: BF89FC07
	s_load_b32 s10, s[0:1], null                               // 00000000166C: F4000280 F8000000
	s_ashr_i32 s3, s2, 31                                      // 000000001674: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001678: BF870499
	s_lshl_b64 s[12:13], s[2:3], 2                             // 00000000167C: 848C8202
	s_add_u32 s6, s6, s12                                      // 000000001680: 80060C06
	s_addc_u32 s7, s7, s13                                     // 000000001684: 82070D07
	s_load_b32 s9, s[6:7], 0x4                                 // 000000001688: F4000243 F8000004
	s_load_b32 s0, s[0:1], 0x4                                 // 000000001690: F4000000 F8000004
	s_waitcnt lgkmcnt(0)                                       // 000000001698: BF89FC07
	v_fma_f32 v0, s8, s10, 0                                   // 00000000169C: D6130000 02001408
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000016A4: BF8704B1
	v_fmac_f32_e64 v0, s9, s0                                  // 0000000016A8: D52B0000 00000009
	v_mov_b32_e32 v1, 0                                        // 0000000016B0: 7E020280
	s_lshl_b64 s[0:1], s[2:3], 2                               // 0000000016B4: 84808202
	s_add_u32 s0, s4, s0                                       // 0000000016B8: 80000004
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000016BC: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 0000000016C0: 20000080
	s_addc_u32 s1, s5, s1                                      // 0000000016C4: 82010105
	global_store_b32 v1, v0, s[0:1]                            // 0000000016C8: DC6A0000 00000001
	s_nop 0                                                    // 0000000016D0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016D4: BFB60003
	s_endpgm                                                   // 0000000016D8: BFB00000
